"""Three memory-access policies over one shared Cue-Tag-Content graph.

  RAGAgent     flat embedding top-k, single shot.
  ActiveAgent  active multi-step reconstruction, no regenerative mechanisms.
               This is an MRAgent-STYLE reimplementation, not MRAgent itself:
               it captures the active/adaptive traversal idea so that the
               comparison isolates REGEN's storage-side contribution, but it
               is not the authors' code and should not be reported as their
               system's score.
  RegenAgent   full REGEN: sigma-gating, predicted depth, P_regen stopping,
               re-seeding, consolidation, fidelity tiers.

Holding the graph and backbone fixed across all three means any difference is
attributable to the access/lifecycle policy rather than to construction.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .graph import MemoryGraph, Params
from .ingest import query_signal
from .llm import LLM
from .signals import SignalVector, sigma as sigma_fn


@dataclass
class Answer:
    text: str
    evidence_dia_ids: list[str] = field(default_factory=list)
    turns: int = 0
    n_materialised: int = 0
    provenance: dict[str, int] = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


ANSWER_RULES = """Answer rules:
- Yes/No questions: answer Yes or No.
- Date/time questions: give the specific date as stated in the evidence.
- Counting questions: give the number.
- Otherwise: give the minimal concrete entity or phrase. Do not write a sentence.
If the evidence does not contain the answer, give your best inference from it.
Never answer "unknown" if any evidence bears on the question."""


def _render(g: MemoryGraph, cids, *, with_tier: bool) -> str:
    lines = []
    for cid in cids:
        c = g.contents.get(cid)
        if not c:
            continue
        body = c.visible_text() if with_tier else c.text
        date = f" [t+{int(c.t_days)}d]" if c.t_days is not None else ""
        ids = f" {{{','.join(c.dia_ids)}}}" if c.dia_ids else ""
        mark = "" if (not with_tier or c.tier == 0) else f" <{c.provenance()}>"
        lines.append(f"- ({c.tag}){date}{ids}{mark} {body}")
    return "\n".join(lines) if lines else "(nothing retrieved yet)"


# ------------------------------------------------------------------ RAG
class RAGAgent:
    name = "RAG"

    def __init__(self, g: MemoryGraph, llm: LLM, k: int = 20):
        self.g, self.llm, self.k = g, llm, k
        self._cids = list(g.contents)
        self._emb = None

    def _ensure(self):
        if self._emb is None:
            texts = [self.g.contents[c].text for c in self._cids]
            self._emb = np.asarray(self.llm.embed(texts, phase="construction_embed"))
            n = np.linalg.norm(self._emb, axis=1, keepdims=True)
            self._emb = self._emb / np.clip(n, 1e-9, None)

    def answer(self, q: str) -> Answer:
        self._ensure()
        qe = np.asarray(self.llm.embed([q], phase="retrieval_embed")[0])
        qe /= max(float(np.linalg.norm(qe)), 1e-9)
        sims = self._emb @ qe
        top = [self._cids[i] for i in np.argsort(-sims)[: self.k]]
        ev = _render(self.g, top, with_tier=False)
        out = self.llm.complete(
            f"{ANSWER_RULES}\n\nEVIDENCE:\n{ev}\n\nQUESTION: {q}\n\nAnswer:",
            phase="retrieval", max_tokens=3000,
        )
        dia = [d for c in top for d in self.g.contents[c].dia_ids]
        return Answer(out.strip(), dia, turns=1, n_materialised=len(top))


# --------------------------------------------------- active reconstruction
NAV_PROMPT = """You are reconstructing a memory to answer a question. Output valid JSON only.

{rules}

QUESTION: {q}

EVIDENCE SO FAR:
{ev}

AVAILABLE ACTIONS:
  "expand_tag"   - pull other episodes sharing the tags of current evidence
  "expand_topic" - pull other episodes under the same topics
  "new_cues"     - search memory for additional keywords you now believe matter

"mode" MUST be exactly "answer" or "navigate" - no other value is valid.
Do NOT add a "reasoning" field and do NOT write prose: think silently, then
emit only the JSON object below. Keep every string on a single line.
If the evidence is sufficient, answer. Otherwise navigate. If you already
navigated twice without progress, answer with your best inference.

{{"mode":"answer","answer":"...","supports":["D1:3"]}}
or
{{"mode":"navigate","actions":["expand_tag"],"new_cues":["..."],"why":"..."}}
"""


class ActiveAgent:
    """MRAgent-style active reconstruction (our reimplementation)."""

    name = "Active (MRAgent-style)"

    def __init__(self, g: MemoryGraph, llm: LLM, max_turns: int = 8, per_turn: int = 20):
        self.g, self.llm = g, llm
        self.max_turns, self.per_turn = max_turns, per_turn

    def _seed(self, q: str):
        _, cues, _ = query_signal(q, self.llm, __import__("datetime").datetime(2023, 1, 1),
                                  fallback_now=None)
        seeds = self.g.seed_by_embedding(q, self.llm, k=self.per_turn)
        seeds |= self.g.seed_by_cues(cues)
        return seeds, cues

    def answer(self, q: str) -> Answer:
        active, cues = self._seed(q)
        H: list[str] = list(active)[: self.per_turn]
        turns = 0
        for turn in range(self.max_turns):
            turns = turn + 1
            ev = _render(self.g, H, with_tier=False)
            try:
                d = self.llm.complete_json(
                    NAV_PROMPT.format(rules=ANSWER_RULES, q=q, ev=ev),
                    phase="retrieval", max_tokens=4000,
                )
            except Exception:  # noqa: BLE001
                break
            if not isinstance(d, dict):
                d = {}
            if d.get("mode") == "answer" and d.get("answer"):
                dia = [x for c in H for x in self.g.contents[c].dia_ids]
                return Answer(str(d["answer"]).strip(), dia, turns, len(H))

            new: set[str] = set()
            acts = d.get("actions") or []
            if "expand_tag" in acts:
                new |= self.g.neighbors_by_tag(H)
            if "expand_topic" in acts:
                new |= self.g.neighbors_by_topic(H)
            if d.get("new_cues"):
                new |= self.g.seed_by_cues([str(x) for x in d["new_cues"]])
            new -= set(H)
            if not new:
                break
            H.extend(list(new)[: self.per_turn])

        ev = _render(self.g, H, with_tier=False)
        out = self.llm.complete(
            f"{ANSWER_RULES}\n\nEVIDENCE:\n{ev}\n\nQUESTION: {q}\n\nAnswer:",
            phase="retrieval", max_tokens=3000,
        )
        dia = [x for c in H for x in self.g.contents[c].dia_ids]
        return Answer(out.strip(), dia, turns, len(H))


# ------------------------------------------------------------------ REGEN
class RegenAgent:
    name = "REGEN"

    def __init__(self, g: MemoryGraph, llm: LLM, epoch, *,
                 consolidate: bool = True, use_gate: bool = True,
                 use_depth: bool = True, use_reseed: bool = True,
                 per_turn: int = 20, max_turns: int = 8):
        self.g, self.llm, self.epoch = g, llm, epoch
        self.consolidate_on = consolidate
        self.use_gate, self.use_depth, self.use_reseed = use_gate, use_depth, use_reseed
        self.per_turn, self.max_turns = per_turn, max_turns

    def _gate(self, cids, qs: SignalVector, theta: float):
        if not self.use_gate:
            return [(c, 1.0) for c in cids]
        return self.g.gate(cids, qs, theta=theta)

    def answer(self, q: str) -> Answer:
        p = self.g.p
        qs, cues, _temporal = query_signal(q, self.llm, self.epoch, fallback_now=self.g.now_days)

        seeds = self.g.seed_by_embedding(q, self.llm, k=self.per_turn)
        seeds |= self.g.seed_by_cues(cues)
        theta = p.theta
        gated = self._gate(seeds, qs, theta)

        # --- predicted depth D = ceil(alpha k / (sigma + beta))  (Eq. 6) ---
        # k_bar must come from the UNGATED seed pool: the gate scores by
        # sigma * gamma^k and therefore selects fresh (low-k) nodes, so
        # estimating k from gate survivors collapses D to 1-2 turns and
        # starves multi-hop reconstruction.
        if seeds:
            pool = list(seeds)
            ks = [self.g.contents[c].stage for c in pool]
            sgs = [
                sigma_fn(qs, self.g.contents[c].signal, self.g.contents[c].stage,
                         delta_0=p.delta_0, eta=p.eta)
                for c in pool
            ]
            k_bar = float(np.median(ks)) if ks else 1.0
            s_bar = float(np.median(sgs)) if sgs else 0.5
        else:
            k_bar, s_bar = 2.0, 0.3
        D = p.predicted_depth(k_bar, s_bar) if self.use_depth else self.max_turns
        D = max(1, min(D, self.max_turns))

        H: list[str] = [c for c, _ in gated[: self.per_turn]]
        turns = 0
        answer_text: str | None = None
        supports: list[str] = []

        prev_p = 0.0
        for turn in range(D):
            turns = turn + 1
            ev = _render(self.g, H, with_tier=True)
            try:
                d = self.llm.complete_json(
                    NAV_PROMPT.format(rules=ANSWER_RULES, q=q, ev=ev),
                    phase="retrieval", max_tokens=4000,
                )
            except Exception:  # noqa: BLE001
                break
            if not isinstance(d, dict):
                d = {}

            if d.get("mode") == "answer" and d.get("answer"):
                answer_text = str(d["answer"]).strip()
                supports = [str(x) for x in (d.get("supports") or [])]
                break

            new: set[str] = set()
            acts = d.get("actions") or []
            if "expand_tag" in acts:
                new |= self.g.neighbors_by_tag(H)
            if "expand_topic" in acts:
                new |= self.g.neighbors_by_topic(H)
            if d.get("new_cues"):
                new |= self.g.seed_by_cues([str(x) for x in d["new_cues"]])
            new -= set(H)

            admitted = [c for c, _ in self._gate(new, qs, theta)][: self.per_turn]

            # --- re-seeding: a failed layer is not the end (Sec. 3.3) ---
            if not admitted and self.use_reseed:
                theta = theta / 2.0
                admitted = [c for c, _ in self._gate(new, qs, theta)][: self.per_turn]
                if not admitted:
                    wider = self.g.neighbors_by_topic(H) | self.g.neighbors_by_tag(H)
                    admitted = [c for c, _ in self._gate(wider - set(H), qs, theta)][: self.per_turn]
            if not admitted:
                break
            H.extend(admitted)

            # --- P_regen stopping rule (Eq. 7) ---
            sig_n = max(
                (sigma_fn(qs, self.g.contents[c].signal, self.g.contents[c].stage,
                          delta_0=p.delta_0, eta=p.eta) for c in H),
                default=0.0,
            )
            k_med = float(np.median([self.g.contents[c].stage for c in H])) if H else 0.0
            pr = 1.0 - (1.0 - min(0.999, sig_n * (p.gamma ** k_med))) ** (turn + 1)
            if pr - prev_p < 0.02 and turn >= 1:
                break
            prev_p = pr

        if answer_text is None:
            ev = _render(self.g, H, with_tier=True)
            answer_text = self.llm.complete(
                f"{ANSWER_RULES}\n\nEVIDENCE:\n{ev}\n\nQUESTION: {q}\n\nAnswer:",
                phase="retrieval", max_tokens=3000,
            ).strip()

        # --- retrieval-induced consolidation (Eq. 8) ---
        prov: dict[str, int] = {}
        for cid in H:
            c = self.g.contents.get(cid)
            if c:
                prov[c.provenance()] = prov.get(c.provenance(), 0) + 1
        if self.consolidate_on:
            used = [c for c in H if any(
                d in supports for d in self.g.contents[c].dia_ids
            )] or H[:5]
            for cid in used:
                self.g.consolidate(cid)

        dia = [x for c in H for x in self.g.contents[c].dia_ids]
        return Answer(answer_text, dia, turns, len(H), prov,
                      {"D": D, "k_bar": k_bar, "s_bar": round(s_bar, 3)})
