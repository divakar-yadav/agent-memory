#!/usr/bin/env python
"""Cued recall of EVICTED evidence — the capability test.

The earlier eviction benchmark sampled questions at random and bucketed them by
age. That was too weak: evidence is duplicated across a median of 8 memories,
so random eviction destroys all copies for only ~3% of questions and the
control was inert.

This version conditions on the thing under test. It selects only questions
whose supporting evidence was ENTIRELY removed by a destructive policy, then
asks every system those same questions:

  full           evidence intact                      -- ceiling
  delete-oldest  evidence deleted outright            -- nothing to recover
  regen          evidence demoted + compressed to gist -- faint but present

The claim being tested is narrow and falsifiable: a compressed gist retains
enough to answer where a deleted memory retains nothing. If regen does not beat
delete-oldest on this subset, non-destructive compression buys nothing.

Usage:  python run_evicted_recall.py -n 40
"""
from __future__ import annotations

import argparse, copy, json, random, time
from datetime import datetime
from pathlib import Path

import anthropic

from regen_mem.graph import MemoryGraph, Params
from regen_mem.llm import LLM, USAGE, _parse_json
from regen_mem.agents import RAGAgent, RegenAgent
from run_eviction import build_timeline, apply_policy, store_bytes, JUDGE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)


def main(a):
    llm = LLM(a.model)
    ac = anthropic.Anthropic()
    P = Params()
    base, qa, now = build_timeline(P)
    budget = int(store_bytes(base) * a.budget)

    # which memories carry each question's evidence
    ev2mem = {}
    for c in base.contents.values():
        for d in c.dia_ids:
            ev2mem.setdefault(d, set()).add(c.cid)

    def staged(policy):
        g = MemoryGraph(P)
        for c in base.contents.values():
            g.add(copy.deepcopy(c))
        apply_policy(g, policy, budget, now)
        return g

    graphs = {p: staged(p) for p in ("full", "delete-oldest", "regen")}
    alive = {p: set(g.contents) for p, g in graphs.items()}

    # KEEP ONLY questions whose evidence delete-oldest destroyed entirely
    sel = []
    for q in qa:
        mems = {m for e in q["evidence"] for m in ev2mem.get(e, ())}
        if not mems:
            continue
        if mems & alive["delete-oldest"]:
            continue                       # evidence survived -- not the test
        if not (mems & alive["regen"]):
            continue                       # regen must still hold it
        tiers = [graphs["regen"].contents[m].tier for m in mems & alive["regen"]]
        sel.append({**q, "_mems": len(mems), "_tier": min(tiers)})
    random.Random(0).shuffle(sel)
    sel = sel[: a.n]

    print(f"timeline {len(base.contents)} memories, budget {a.budget:.0%}")
    print(f"questions whose evidence delete-oldest ERASED and regen still holds: {len(sel)}")
    import statistics as st
    print(f"  median evidence redundancy {st.median([q['_mems'] for q in sel]):.0f} memories")
    print(f"  regen fidelity of that evidence: "
          f"{st.mean([q['_tier'] for q in sel]):.2f} mean tier (0=verbatim, 3=tag only)\n")

    def judge(q, gold, pred):
        if not pred.strip():
            return False
        r = ac.messages.create(model="claude-haiku-4-5", max_tokens=200,
            messages=[{"role": "user", "content": JUDGE.format(q=q, gold=gold, pred=pred)},
                      {"role": "assistant", "content": "{"}])
        d = _parse_json("{" + "".join(b.text for b in r.content if b.type == "text"))
        return bool(d.get("correct")) if isinstance(d, dict) else False

    rows = []
    for policy in ("full", "delete-oldest", "regen"):
        g = graphs[policy]
        agent = (RegenAgent(g, llm, datetime(2023, 1, 1), consolidate=False)
                 if policy == "regen" else RAGAgent(g, llm, k=20))
        USAGE.reset(); ok = 0
        for i, q in enumerate(sel):
            try:
                pred = agent.answer(q["question"]).text
            except Exception as e:
                print(f"    ! q{i}: {type(e).__name__}: {str(e)[:80]}"); pred = ""
            good = judge(q["question"], str(q["answer"]), pred)
            ok += good
            rows.append({"policy": policy, "q": q["question"], "gold": str(q["answer"]),
                         "pred": pred, "correct": good, "tier": q["_tier"],
                         "redundancy": q["_mems"], "category": q.get("category")})
            json.dump(rows, open(RESULTS / "evicted_recall.json", "w"), indent=1)
            print(f"  {policy:14s} {i+1}/{len(sel)} acc={ok/(i+1):.0%}", end="\r")
        u = USAGE.snapshot()
        print(f"  {policy:14s} acc={100*ok/len(sel):5.1f}%   tok/q={u['total_tokens']//len(sel):,}      ")

    print("\n" + "=" * 58)
    print("ACCURACY ON QUESTIONS WHOSE EVIDENCE WAS EVICTED")
    print("-" * 58)
    for policy in ("full", "delete-oldest", "regen"):
        rs = [r for r in rows if r["policy"] == policy]
        note = {"full": "evidence intact (ceiling)",
                "delete-oldest": "evidence deleted",
                "regen": "evidence compressed to gist"}[policy]
        print(f"  {policy:14s} {100*sum(r['correct'] for r in rs)/len(rs):5.1f}%   {note}")
    print("=" * 58)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--budget", type=float, default=0.35)
    ap.add_argument("-n", type=int, default=40)
    main(ap.parse_args())
