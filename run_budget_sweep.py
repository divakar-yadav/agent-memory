#!/usr/bin/env python
"""Budget sweep for recall-after-eviction, plus a stronger deletion baseline.

Two reviewer objections to the single-point result:
  1. one budget is not a curve
  2. delete-oldest is a weak policy, trivially beaten

This addresses both. It sweeps the capacity budget and adds delete-redundant,
which evicts the memories most similar to a surviving neighbour first. Since
evidence is duplicated across a median of 8 memories, shedding duplicates is
what a competent destructive policy would actually do, and it preserves
distinct information far better than recency ordering.

At each budget the evicted question set is recomputed, because which evidence
is destroyed depends on the budget.

Usage:  python run_budget_sweep.py --budgets 0.25,0.35,0.50,0.65 -n 25
"""
from __future__ import annotations

import argparse, copy, json, random
from datetime import datetime
from pathlib import Path

import numpy as np
import anthropic

from regen_mem.graph import MemoryGraph, Params
from regen_mem.llm import LLM, USAGE, _parse_json
from regen_mem.agents import RAGAgent, RegenAgent
from run_eviction import build_timeline, apply_policy, store_bytes, JUDGE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)


def delete_redundant(g: MemoryGraph, budget: int, llm: LLM) -> dict:
    """Evict the most-duplicated memories first.

    Rank every memory by cosine similarity to its nearest neighbour; drop the
    highest first, so near-duplicates go before unique content. This is the
    policy a system that knows its store is redundant would use.
    """
    cids = list(g.contents)
    E = np.asarray(llm.embed([g.contents[c].text for c in cids],
                             phase="construction_embed"), dtype=np.float32)
    E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    nearest = S.max(axis=1)                      # similarity to closest neighbour
    order = [cids[i] for i in np.argsort(-nearest)]   # most redundant first
    dropped = 0
    for cid in order:
        if store_bytes(g) <= budget:
            break
        del g.contents[cid]; dropped += 1
    for idx in (g.cue_index, g.tag_index, g.topic_index):
        for k in list(idx):
            idx[k] = {c for c in idx[k] if c in g.contents}
    return {"kept": len(g.contents), "deleted": dropped, "bytes": store_bytes(g)}


def main(a):
    llm = LLM(a.model); ac = anthropic.Anthropic(); P = Params()
    base, qa, now = build_timeline(P)
    full_bytes = store_bytes(base)
    ev2mem = {}
    for c in base.contents.values():
        for d in c.dia_ids:
            ev2mem.setdefault(d, set()).add(c.cid)

    def judge(q, gold, pred):
        if not pred.strip():
            return False
        r = ac.messages.create(model="claude-haiku-4-5", max_tokens=200,
            messages=[{"role": "user", "content": JUDGE.format(q=q, gold=gold, pred=pred)},
                      {"role": "assistant", "content": "{"}])
        d = _parse_json("{" + "".join(b.text for b in r.content if b.type == "text"))
        return bool(d.get("correct")) if isinstance(d, dict) else False

    def stage(policy, budget):
        g = MemoryGraph(P)
        for c in base.contents.values():
            g.add(copy.deepcopy(c))
        if policy == "delete-redundant":
            g.apply_decay(now); delete_redundant(g, budget, llm)
        else:
            apply_policy(g, policy, budget, now)
        return g

    rows = []
    print(f"timeline {len(base.contents)} memories, {full_bytes:,} B, "
          f"{now:.0f} days\n")
    for frac in [float(x) for x in a.budgets.split(",")]:
        budget = int(full_bytes * frac)
        graphs = {p: stage(p, budget) for p in
                  ("full", "delete-oldest", "delete-redundant", "regen")}
        alive = {p: set(g.contents) for p, g in graphs.items()}

        # questions whose evidence BOTH destructive policies erased, and which
        # regen still holds -- the set where the capability claim is testable
        sel = []
        for q in qa:
            mems = {m for e in q["evidence"] for m in ev2mem.get(e, ())}
            if not mems or not (mems & alive["regen"]):
                continue
            if (mems & alive["delete-oldest"]) or (mems & alive["delete-redundant"]):
                continue
            sel.append(q)
        random.Random(0).shuffle(sel); sel = sel[: a.n]
        if len(sel) < 6:
            print(f"budget {frac:.0%}: only {len(sel)} questions fully evicted "
                  f"by both policies -- skipping (not enough signal)\n")
            continue

        print(f"budget {frac:.0%} = {budget:,} B   evicted-by-both questions: {len(sel)}")
        line = {}
        for policy in ("full", "delete-oldest", "delete-redundant", "regen"):
            g = graphs[policy]
            agent = (RegenAgent(g, llm, datetime(2023, 1, 1), consolidate=False)
                     if policy == "regen" else RAGAgent(g, llm, k=20))
            ok = 0
            for i, q in enumerate(sel):
                try:
                    pred = agent.answer(q["question"]).text
                except Exception:
                    pred = ""
                good = judge(q["question"], str(q["answer"]), pred)
                ok += good
                rows.append({"budget": frac, "policy": policy, "q": q["question"],
                             "gold": str(q["answer"]), "pred": pred, "correct": good})
                json.dump(rows, open(RESULTS / "budget_sweep.json", "w"), indent=1)
                print(f"    {policy:17s} {i+1}/{len(sel)}", end="\r")
            line[policy] = 100 * ok / len(sel)
            print(f"    {policy:17s} {line[policy]:5.1f}%                ")
        print()

    print("=" * 74)
    print(f"{'budget':>8s}{'full':>12s}{'del-oldest':>14s}{'del-redundant':>16s}{'graded':>12s}")
    print("-" * 74)
    for frac in sorted({r["budget"] for r in rows}):
        cells = ""
        for p in ("full", "delete-oldest", "delete-redundant", "regen"):
            rs = [r for r in rows if r["budget"] == frac and r["policy"] == p]
            cells += f"{100*sum(r['correct'] for r in rs)/len(rs):11.1f}%" if rs else f"{'-':>12s}"
        print(f"{frac:7.0%}{cells}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--budgets", default="0.25,0.35,0.50")
    ap.add_argument("-n", type=int, default=25)
    main(ap.parse_args())
