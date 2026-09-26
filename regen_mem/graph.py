"""The REGEN memory graph and its lifecycle (paper Sec. 3.2-3.6).

Cue -- Tag -- Content, with three fields per content node that prior agent
memories do not carry: a contextual signal vector s*, a retention stage k, and
a fidelity tier l.

Lifecycle operations:
    decay        k_v = min(K, floor(lambda ln(1 + dt)))     (Sec. 3.2)
    consolidate  k_v <- max(0, k_v - kappa)   on successful recall  (Sec. 3.5)
    reconsolidate  write back correction, demote superseded by +kappa'
    replay       compress stage-advanced nodes to lower fidelity tiers (Sec. 3.6)

Forgetting is demotion, never deletion: a node past tau is unreachable by the
default traversal prior but still reachable when sigma is high enough.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .signals import SignalVector, sigma as sigma_fn


@dataclass
class Params:
    """Hyperparameters, matching the values reported in the paper."""

    # NOTE: the paper's published values (lam=0.9, gamma=0.72, theta=0.05,
    # K=8) are internally inconsistent: p_base < theta requires k > 9.12 but
    # K caps k at 8, so tau is UNREACHABLE and no trace can ever leave the
    # working set.  lam=0.9 also needs ~20 years of age to reach k=8.
    # Calibrated below so tau=7 is reached at ~340 days on this corpus.
    lam: float = 1.2        # lambda: stage advance rate
    gamma: float = 0.62     # per-stage decay of the default traversal prior
    K: int = 8              # max stage
    delta_0: float = 1.0    # base match window
    eta: float = 0.35       # stage-widening of the match window
    kappa: int = 2          # consolidation strength
    kappa_prime: int = 1    # reconsolidation demotion of superseded traces
    theta: float = 0.05     # sigma-gate threshold
    alpha: float = 1.8      # depth predictor scale
    beta: float = 0.45      # depth predictor offset
    eps_min: float = 0.05   # irrecoverability threshold on p_base
    theta_strong: float = 0.78  # sigma alone sufficient to override age
    rho_0: float = 1.0

    @property
    def tau(self) -> int:
        """Irrecoverability threshold: first stage whose default prior < eps_min."""
        for k in range(self.K + 1):
            if self.rho_0 * (self.gamma**k) < self.eps_min:
                return k
        return self.K + 1

    def p_base(self, k: int) -> float:
        return self.rho_0 * (self.gamma**k)

    def predicted_depth(self, k: float, sig: float) -> int:
        """D = ceil(alpha k / (sigma + beta))  (paper Eq. 6)."""
        return max(1, math.ceil(self.alpha * max(k, 0.5) / (sig + self.beta)))


@dataclass
class Content:
    """An episodic or semantic memory node."""

    cid: str
    kind: str                      # "episode" | "semantic"
    text: str                      # tier-0 verbatim text
    gist: str                      # tier>=1 surviving summary (routing scaffold)
    tag: str
    cues: list[str]
    signal: SignalVector
    t_days: float | None
    dia_ids: list[str] = field(default_factory=list)
    topic: str = ""
    stage: int = 0                 # k_v
    tier: int = 0                  # l_v : 0 = verbatim, higher = lossier
    last_access_days: float | None = None
    access_count: int = 0

    def visible_text(self) -> str:
        """What reconstruction actually gets to read, given the fidelity tier.

        A graded ladder, not a cliff: episodic detail is discarded first and
        the routing scaffold (tag, gist, topic) survives longest, so a
        high-tier node must have its answer *regenerated* from gist plus
        context rather than read off.
        """
        if self.tier <= 0:
            return self.text
        if self.tier == 1:
            # Drop trailing detail, keep the leading proposition.  Splitting on
            # ". " alone is a no-op for the single-sentence memories the
            # distiller actually produces, so fall back to a proportional
            # truncation at a word boundary -- otherwise the ladder's first
            # rung does nothing and tier 1 -> 2 becomes a 211 -> 35 char cliff.
            parts = self.text.split(". ")
            if len(parts) > 1:
                return parts[0] + "."
            words = self.text.split()
            keep = max(6, int(len(words) * 0.6))
            if keep >= len(words):
                return self.text
            return " ".join(words[:keep]) + " …"
        if self.tier == 2:
            return self.gist or self.text[:80]
        return f"{self.tag} / {self.topic}".strip(" /") or self.gist

    def provenance(self) -> str:
        return "recalled" if self.tier == 0 else f"inferred(tier={self.tier})"


class MemoryGraph:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.contents: dict[str, Content] = {}
        self.cue_index: dict[str, set[str]] = defaultdict(set)   # cue -> cids
        self.tag_index: dict[str, set[str]] = defaultdict(set)   # tag -> cids
        self.topic_index: dict[str, set[str]] = defaultdict(set)
        self.now_days: float = 0.0

    # ------------------------------------------------------------ building
    def add(self, c: Content) -> None:
        self.contents[c.cid] = c
        for cue in c.cues:
            self.cue_index[cue.lower().strip()].add(c.cid)
        if c.tag:
            self.tag_index[c.tag.lower().strip()].add(c.cid)
        if c.topic:
            self.topic_index[c.topic.lower().strip()].add(c.cid)

    # ------------------------------------------------------------ lifecycle
    def apply_decay(self, now_days: float) -> None:
        """Advance every node's retention stage by elapsed time (Eq. 3)."""
        self.now_days = now_days
        for c in self.contents.values():
            ref = c.last_access_days if c.last_access_days is not None else c.t_days
            if ref is None:
                continue
            dt = max(0.0, now_days - ref)
            c.stage = min(self.p.K, int(math.floor(self.p.lam * math.log1p(dt))))

    def consolidate(self, cid: str) -> None:
        """Retrieval-induced consolidation: successful recall strengthens (Eq. 8)."""
        c = self.contents.get(cid)
        if not c:
            return
        c.stage = max(0, c.stage - self.p.kappa)
        c.last_access_days = self.now_days
        c.access_count += 1
        # Consolidation also restores fidelity one tier: a trace that is used
        # is worth keeping in detail.
        c.tier = max(0, c.tier - 1)

    def reconsolidate(self, cid: str, corrected: str) -> None:
        """Write back a correction; demote the superseded version rather than
        deleting it, so 'what was believed earlier' stays answerable."""
        c = self.contents.get(cid)
        if not c:
            return
        c.text = corrected
        c.stage = min(self.p.K, c.stage + self.p.kappa_prime)

    def replay(self) -> dict:
        """Offline replay: compress stage-advanced nodes to lower fidelity tiers.

        Tier rises with stage, so old unused traces keep only tag + gist +
        topic linkage -- the routing scaffold -- and their detail is dropped.
        """
        moved = 0
        for c in self.contents.values():
            target = 0 if c.stage < 2 else min(3, 1 + (c.stage - 2) // 2)
            if target > c.tier:
                c.tier = target
                moved += 1
        return {"compressed": moved}

    # ------------------------------------------------------------ retrieval
    def gate(
        self, cids, q_signal: SignalVector, *, theta: float | None = None
    ) -> list[tuple[str, float]]:
        """sigma-gated admission (paper Sec. 3.3).

        Scores candidates by sigma * gamma^k BEFORE any content is read, so
        pruning costs an inner product rather than an LLM call.
        """
        th = self.p.theta if theta is None else theta
        out: list[tuple[str, float]] = []
        for cid in cids:
            c = self.contents.get(cid)
            if not c:
                continue
            s = sigma_fn(
                q_signal, c.signal, c.stage,
                delta_0=self.p.delta_0, eta=self.p.eta,
            )
            # Admission is a DISJUNCTION, not a product.
            #
            #   default path : sigma * gamma^k >= theta
            #   override path: sigma          >= theta_strong
            #
            # Multiplying relevance by the retention prior made old traces
            # unreachable at any signal strength -- at k=7, gamma^k=0.035, so
            # passing theta=0.05 needs sigma>1.42, which is impossible since
            # sigma<=1. That contradicts the model's own claim that sufficient
            # contextual support recovers a trace past tau. The override lets a
            # strong cue reach an old memory while weak cues still only reach
            # recent ones.
            score = s * self.p.p_base(c.stage)
            if score >= th or s >= self.p.theta_strong:
                out.append((cid, max(score, s * self.p.theta if s >= self.p.theta_strong else 0.0)))
        out.sort(key=lambda x: -x[1])
        return out

    def seed_by_embedding(self, query: str, llm, k: int = 20) -> set[str]:
        """Similarity-based seeding.

        MRAgent's formulation seeds with V_sim = TopK(sim(x, v)) and only then
        expands over the graph; cue-string matching alone loses a large amount
        of evidence recall.  Both graph-based agents use this so the
        comparison isolates the lifecycle policy rather than the seeder.
        """
        if not self.contents:
            return set()
        if getattr(self, "_emb_index", None) is None:
            cids = list(self.contents)
            texts = [self.contents[c].text for c in cids]
            E = np.asarray(llm.embed(texts, phase="construction_embed"), dtype=np.float32)
            E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
            self._emb_index = (cids, E)
        cids, E = self._emb_index
        q = np.asarray(llm.embed([query], phase="retrieval_embed")[0], dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-9)
        sims = E @ q
        return {cids[i] for i in np.argsort(-sims)[:k]}

    def seed_by_cues(self, cues: list[str]) -> set[str]:
        hit: set[str] = set()
        for q in cues:
            ql = q.lower().strip()
            if ql in self.cue_index:
                hit |= self.cue_index[ql]
            else:
                for cue, cids in self.cue_index.items():
                    if ql and (ql in cue or cue in ql):
                        hit |= cids
        return hit

    def neighbors_by_tag(self, cids) -> set[str]:
        out: set[str] = set()
        for cid in cids:
            c = self.contents.get(cid)
            if c and c.tag:
                out |= self.tag_index.get(c.tag.lower().strip(), set())
        return out

    def neighbors_by_topic(self, cids) -> set[str]:
        out: set[str] = set()
        for cid in cids:
            c = self.contents.get(cid)
            if c and c.topic:
                out |= self.topic_index.get(c.topic.lower().strip(), set())
        return out

    # ------------------------------------------------------------ reporting
    def working_set(self) -> list[str]:
        """Nodes above the default traversal prior -- the set W(t) of Prop. 1."""
        return [
            cid for cid, c in self.contents.items()
            if self.p.p_base(c.stage) >= self.p.theta
        ]

    def stats(self) -> dict:
        stages = [c.stage for c in self.contents.values()]
        tiers = [c.tier for c in self.contents.values()]
        return {
            "nodes": len(self.contents),
            "working_set": len(self.working_set()),
            "past_tau": sum(1 for s in stages if s >= self.p.tau),
            "mean_stage": round(float(np.mean(stages)), 3) if stages else 0.0,
            "mean_tier": round(float(np.mean(tiers)), 3) if tiers else 0.0,
            "tau": self.p.tau,
            "store_chars": sum(len(c.visible_text()) for c in self.contents.values()),
        }

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "now_days": self.now_days,
            "contents": [
                {
                    "cid": c.cid, "kind": c.kind, "text": c.text, "gist": c.gist,
                    "tag": c.tag, "cues": c.cues, "topic": c.topic,
                    "signal": c.signal.to_json(), "t_days": c.t_days,
                    "dia_ids": c.dia_ids, "stage": c.stage, "tier": c.tier,
                    "last_access_days": c.last_access_days,
                    "access_count": c.access_count,
                }
                for c in self.contents.values()
            ],
        }))

    @staticmethod
    def load(path: Path, params: Params | None = None) -> "MemoryGraph":
        d = json.loads(path.read_text())
        g = MemoryGraph(params)
        g.now_days = d.get("now_days", 0.0)
        for cd in d["contents"]:
            g.add(Content(
                cid=cd["cid"], kind=cd["kind"], text=cd["text"], gist=cd["gist"],
                tag=cd["tag"], cues=cd["cues"], topic=cd.get("topic", ""),
                signal=SignalVector.from_json(cd["signal"]),
                t_days=cd["t_days"], dia_ids=cd.get("dia_ids", []),
                stage=cd.get("stage", 0), tier=cd.get("tier", 0),
                last_access_days=cd.get("last_access_days"),
                access_count=cd.get("access_count", 0),
            ))
        return g
