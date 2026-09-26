#!/usr/bin/env python
"""The three experiments the REGEN architecture actually stands on.

  --continual   store size grows, question set held fixed  (paper Sec. 5.4)
  --aging       recall vs. age of supporting evidence      (paper Sec. 5.5)
  --recovery    recovery for traces past tau               (paper Sec. 5.6)

Continual replay is simulated by padding the memory graph with distractor
nodes drawn from OTHER conversations, so the candidate set grows while the
evaluated questions and their evidence stay identical.  That isolates
candidate-set dilution from question difficulty, which is the whole point.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from regen_mem.agents import ActiveAgent, RegenAgent
from regen_mem.graph import Content, MemoryGraph, Params
from regen_mem.ingest import build_graph, parse_locomo_date
from regen_mem.llm import LLM, USAGE
from regen_mem.signals import sigma as sigma_fn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "locomo10.json"
CACHE = ROOT / ".graphs"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

from run_benchmark import CATEGORY, get_graph, judge  # reuse harness pieces


def load_or_build(sample, llm, params):
    return get_graph(sample, llm, params, None)


def answerable(sample, g, n, seed=0):
    kept = {d for c in g.contents.values() for d in c.dia_ids}
    qa = [q for q in sample["qa"]
          if q.get("category") != 5 and q.get("evidence")
          and all(e in kept for e in q["evidence"])]
    random.Random(seed).shuffle(qa)
    return qa[:n]


def pad_graph(base: MemoryGraph, donors: list[MemoryGraph], mult: int, params: Params) -> MemoryGraph:
    """Grow the store mult-fold with distractors from other conversations.

    Evidence nodes are untouched; only the haystack grows.
    """
    g = MemoryGraph(params)
    for c in base.contents.values():
        g.add(copy.deepcopy(c))
    if mult <= 1:
        g.now_days = base.now_days
        return g
    need = len(base.contents) * (mult - 1)
    pool: list[Content] = [c for d in donors for c in d.contents.values()]
    rng = random.Random(7)
    i = 0
    # Distractors represent history accumulated BEFORE the evaluated
    # conversation, so they are dated into the past.  This is what a
    # long-running agent's store actually looks like: a large tail of old
    # material plus a small recent working set.  Gold evidence is untouched.
    while need > 0 and pool:
        src = pool[rng.randrange(len(pool))]
        c = copy.deepcopy(src)
        c.cid = f"pad{i}_{c.cid}"
        c.dia_ids = []            # distractors carry no gold evidence ids
        age_back = rng.uniform(200, 1800)   # days before the corpus epoch
        c.t_days = -age_back
        c.signal.t_days = -age_back
        c.last_access_days = None
        g.add(c)
        i += 1
        need -= 1
    g.now_days = base.now_days
    return g


# ------------------------------------------------------------------ continual
def continual(args, llm, judge_llm, params):
    samples = json.loads(DATA.read_text())
    base_s = samples[0]
    g0, epoch = load_or_build(base_s, llm, params)
    donors = []
    for s in samples[1:4]:
        gd, _ = load_or_build(s, llm, params)
        donors.append(gd)

    qa = answerable(base_s, g0, args.questions)
    last_t = max((c.t_days or 0) for c in g0.contents.values())
    print(f"continual: {len(qa)} questions, base graph {len(g0.contents)} nodes\n")

    rows = []
    for mult in [1, 4, 16]:
        for name in ["Active (MRAgent-style)", "REGEN"]:
            g = pad_graph(g0, donors, mult, params)
            g.apply_decay(last_t)
            if name == "REGEN":
                g.replay()
                agent = RegenAgent(g, llm, epoch)
            else:
                agent = ActiveAgent(g, llm)
            st = g.stats()
            USAGE.reset()
            t0 = time.time()
            ok = 0
            for q in qa:
                a = agent.answer(q["question"])
                ok += judge(judge_llm, q["question"], str(q["answer"]), a.text)
            u = USAGE.snapshot()
            row = {
                "experiment": "continual", "mult": mult, "system": name,
                "nodes": st["nodes"], "working_set": st["working_set"],
                "store_chars": st["store_chars"],
                "acc": 100 * ok / len(qa),
                "tokens_per_q": u["total_tokens"] / len(qa),
                "seconds": time.time() - t0,
            }
            rows.append(row)
            print(f"  {mult:2d}x  {name:24s} nodes={st['nodes']:5d} "
                  f"W={st['working_set']:5d} acc={row['acc']:5.1f}% "
                  f"tok/q={row['tokens_per_q']:8,.0f}")
    return rows


# --------------------------------------------------------------------- aging
def aging(args, llm, judge_llm, params):
    """Recall as a function of the age of the supporting evidence.

    We sweep the query time forward, which ages every trace, and record
    evidence recall.  A staircase indicates staged decay; a smooth curve
    indicates continuous decay.
    """
    samples = json.loads(DATA.read_text())
    s = samples[0]
    g0, epoch = load_or_build(s, llm, params)
    qa = answerable(s, g0, args.questions)
    last_t = max((c.t_days or 0) for c in g0.contents.values())

    rows = []
    for extra in [0, 15, 45, 120, 300, 700, 1500]:
        now = last_t + extra
        g = MemoryGraph.load(CACHE / f"{s.get('sample_id')}_None.json", params)
        g.apply_decay(now)
        g.replay()
        st = g.stats()
        agent = RegenAgent(g, llm, epoch, consolidate=False)
        USAGE.reset()
        recs, oks = [], 0
        for q in qa:
            a = agent.answer(q["question"])
            gold = q.get("evidence", [])
            got = set(a.evidence_dia_ids)
            recs.append(sum(1 for x in gold if x in got) / max(1, len(gold)))
            oks += judge(judge_llm, q["question"], str(q["answer"]), a.text)
        row = {
            "experiment": "aging", "extra_days": extra,
            "mean_stage": st["mean_stage"], "mean_tier": st["mean_tier"],
            "working_set": st["working_set"], "past_tau": st["past_tau"],
            "ev_recall": float(np.mean(recs)), "acc": 100 * oks / len(qa),
        }
        rows.append(row)
        print(f"  +{extra:5d}d  k={st['mean_stage']:.2f} tier={st['mean_tier']:.2f} "
              f"W={st['working_set']:4d} past_tau={st['past_tau']:4d} "
              f"recall={row['ev_recall']:.3f} acc={row['acc']:5.1f}%")
    return rows


# ------------------------------------------------------------------ recovery
def recovery(args, llm, judge_llm, params):
    """Recovery for traces demoted past tau, as a function of signal fidelity.

    Deletion (what Mem0 does) is recoverable with probability exactly zero.
    Demotion is not: we measure what fraction of gold evidence is still
    reached when every relevant trace has been pushed past tau.
    """
    samples = json.loads(DATA.read_text())
    s = samples[0]
    g0, epoch = load_or_build(s, llm, params)
    qa = answerable(s, g0, args.questions)

    rows = []
    for mode, theta_mult in [("default gate", 1.0), ("relaxed gate (re-seed)", 0.05)]:
        g = MemoryGraph.load(CACHE / f"{s.get('sample_id')}_None.json", params)
        # Force every node past the irrecoverability threshold.
        for c in g.contents.values():
            c.stage = params.tau
        g.replay()
        g.p.theta = params.theta * theta_mult
        agent = RegenAgent(g, llm, epoch, consolidate=False, use_reseed=(theta_mult < 1))
        recs, oks = [], 0
        for q in qa:
            a = agent.answer(q["question"])
            gold = q.get("evidence", [])
            got = set(a.evidence_dia_ids)
            recs.append(sum(1 for x in gold if x in got) / max(1, len(gold)))
            oks += judge(judge_llm, q["question"], str(q["answer"]), a.text)
        row = {
            "experiment": "recovery", "mode": mode,
            "theta": g.p.theta, "stage": params.tau,
            "ev_recall": float(np.mean(recs)), "acc": 100 * oks / len(qa),
        }
        rows.append(row)
        print(f"  k=tau={params.tau}  {mode:24s} theta={g.p.theta:.4f} "
              f"recall={row['ev_recall']:.3f} acc={row['acc']:5.1f}%")

    rows.append({"experiment": "recovery", "mode": "deletion (Mem0-style)",
                 "ev_recall": 0.0, "acc": 0.0,
                 "note": "deleted traces are unrecoverable by construction"})
    print(f"  {'deletion (Mem0-style)':38s} recall=0.000 acc=  0.0%  (by construction)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--questions", type=int, default=20)
    ap.add_argument("--continual", action="store_true")
    ap.add_argument("--aging", action="store_true")
    ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--out", default="experiments.json")
    a = ap.parse_args()

    llm, jl, P = LLM(a.model), LLM(a.judge_model), Params()
    out = []
    if a.continual:
        print("\n=== CONTINUAL REPLAY ===");  out += continual(a, llm, jl, P)
    if a.aging:
        print("\n=== AGE-STRATIFIED RECALL ===");  out += aging(a, llm, jl, P)
    if a.recovery:
        print("\n=== RECOVERY PAST TAU ===");  out += recovery(a, llm, jl, P)
    (RESULTS / a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote results/{a.out}")
