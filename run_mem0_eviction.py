#!/usr/bin/env python
"""Mem0 under a capacity bound, against graded forgetting at the same bound.

Mem0 is the strongest baseline in this study and was absent from the eviction
benchmark. Adding it requires confronting a fact about the system: mem0 has no
capacity policy. Its reconciler issued 381 ADD and 0 DELETE on this corpus, so
the store grows linearly and simply overflows any budget. To put it under one
we must bolt on an eviction rule; the only mechanism mem0 itself provides is
memories.delete, so we evict oldest-first by created_at (which follows session
chronology, since ingestion was sequential).

Mem0 memories carry no dialogue ids, so the evicted-evidence protocol of
run_evicted_recall.py cannot be applied to it. Instead both systems are held to
the SAME ABSOLUTE byte budget on the same corpus and asked the same questions,
stratified by the age of their supporting evidence. This is the weaker protocol
but it is the one both representations admit.

Usage:  python run_mem0_eviction.py --budget 0.35 -n 30
"""
from __future__ import annotations

import argparse, copy, json, random, re
from datetime import datetime
from pathlib import Path

import anthropic

from regen_mem.graph import MemoryGraph, Params
from regen_mem.llm import LLM, _parse_json
from regen_mem.agents import RegenAgent, RAGAgent, ANSWER_RULES
from regen_mem.ingest import parse_locomo_date
from run_eviction import JUDGE, store_bytes

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
SID = "conv-26"


def evidence_ages():
    s = next(x for x in json.loads((ROOT / "data" / "locomo10.json").read_text())
             if x["sample_id"] == SID)
    conv = s["conversation"]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    dates = {k: parse_locomo_date(conv.get(f"{k}_date_time", "")) for k in keys}
    ep = min(d for d in dates.values() if d)
    day = {t["dia_id"]: (dates[k] - ep).days
           for k in keys for t in (conv.get(k) or []) if t.get("dia_id") and dates[k]}
    last = max(day.values())
    return s, day, last


def main(a):
    llm = LLM("claude-sonnet-4-5")
    ac = anthropic.Anthropic()
    P = Params()

    sample, day, last = evidence_ages()
    ours = MemoryGraph.load(ROOT / ".graphs" / f"{SID}_None_claude-sonnet-4-5.json", P)
    full_ours = store_bytes(ours)

    from run_mem0 import make_memory, USER_ID
    mem = make_memory()
    m_all = mem.get_all(filters={"user_id": USER_ID}, top_k=5000).get("results", [])
    full_mem0 = sum(len(r.get("memory", "")) for r in m_all)

    # one absolute cap both stores must fit under
    budget = int(min(full_ours, full_mem0) * a.budget)
    print(f"full stores : ours {full_ours:,} B | mem0 {full_mem0:,} B ({len(m_all)} memories)")
    print(f"shared cap  : {budget:,} B  -> ours sheds "
          f"{100*(1-budget/full_ours):.0f}%, mem0 sheds {100*(1-budget/full_mem0):.0f}%\n")

    # ---- mem0: evict oldest-first until under cap (its only mechanism) ----
    ordered = sorted(m_all, key=lambda r: r.get("created_at") or "")
    keep, acc = [], 0
    for r in reversed(ordered):                     # newest first, keep while it fits
        c = len(r.get("memory", ""))
        if acc + c > budget:
            continue
        keep.append(r); acc += c
    kept_ids = {r["id"] for r in keep}
    print(f"mem0 after eviction : {len(keep)}/{len(m_all)} memories, {acc:,} B "
          f"({len(m_all)-len(keep)} deleted)")

    # ---- ours: compress to the same cap, delete nothing ----
    g = MemoryGraph(P)
    for c in ours.contents.values():
        g.add(copy.deepcopy(c))
    g.apply_decay(last)
    for c in g.contents.values():
        c.tier = 0
    order = sorted(g.contents.values(), key=lambda c: -(last - (c.t_days or 0)))
    i = 0
    while store_bytes(g) > budget and i < len(order) * 3:
        cc = order[i % len(order)]
        if cc.tier < 3:
            cc.tier += 1
        i += 1
    print(f"ours after compression: {len(g.contents)}/{len(ours.contents)} memories, "
          f"{store_bytes(g):,} B (0 deleted)\n")

    kept_dia = {d for c in ours.contents.values() for d in c.dia_ids}
    qa = [q for q in sample["qa"] if q.get("category") != 5 and q.get("evidence")
          and all(e in kept_dia for e in q["evidence"])]
    for q in qa:
        ages = [day[e] for e in q["evidence"] if e in day]
        q["_age"] = last - min(ages) if ages else 0
    random.Random(0).shuffle(qa); qa = qa[: a.n]

    def judge(q, gold, pred):
        if not pred.strip():
            return False
        r = ac.messages.create(model="claude-haiku-4-5", max_tokens=200,
            messages=[{"role": "user", "content": JUDGE.format(q=q, gold=gold, pred=pred)},
                      {"role": "assistant", "content": "{"}])
        d = _parse_json("{" + "".join(b.text for b in r.content if b.type == "text"))
        return bool(d.get("correct")) if isinstance(d, dict) else False

    rows = []
    for name in ("mem0 (evicted)", "graded (compressed)"):
        ok = 0
        agent = RegenAgent(g, llm, datetime(2023, 1, 1), consolidate=False) \
            if name.startswith("graded") else None
        for i, q in enumerate(qa):
            if agent is not None:
                pred = agent.answer(q["question"]).text
            else:
                hits = mem.search(query=q["question"], filters={"user_id": USER_ID}, top_k=20)
                res = [h for h in hits.get("results", []) if h["id"] in kept_ids]
                ev = "\n".join(f"- {h.get('memory','')}" for h in res) or "(nothing retrieved)"
                r = ac.messages.create(model="claude-sonnet-4-5", max_tokens=800,
                    messages=[{"role": "user", "content":
                        f"{ANSWER_RULES}\n\nEVIDENCE:\n{ev}\n\nQUESTION: {q['question']}\n\nAnswer:"}])
                pred = "".join(b.text for b in r.content if b.type == "text").strip()
            good = judge(q["question"], str(q["answer"]), pred)
            ok += good
            rows.append({"system": name, "q": q["question"], "gold": str(q["answer"]),
                         "pred": pred, "correct": good, "age": q["_age"]})
            json.dump(rows, open(RESULTS / "mem0_eviction.json", "w"), indent=1)
            print(f"  {name:21s} {i+1}/{len(qa)} acc={ok/(i+1):.0%}", end="\r")
        print(f"  {name:21s} acc={100*ok/len(qa):5.1f}%              ")

    B = [("recent 0-60d", 0, 60), ("mid 61-120d", 61, 120), ("old 121d+", 121, 10**9)]
    print("\n" + "=" * 66)
    print(f"{'system':22s}" + "".join(f"{n:>14s}" for n, _, _ in B) + f"{'ALL':>8s}")
    print("-" * 66)
    for name in ("mem0 (evicted)", "graded (compressed)"):
        rs = [r for r in rows if r["system"] == name]
        line = f"{name:22s}"
        for n, lo, hi in B:
            sub = [r for r in rs if lo <= r["age"] <= hi]
            line += f"{100*sum(r['correct'] for r in sub)/len(sub):13.1f}%" if sub else f"{'-':>14s}"
        line += f"{100*sum(r['correct'] for r in rs)/len(rs):7.1f}%"
        print(line)
    print("=" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.35)
    ap.add_argument("-n", type=int, default=30)
    main(ap.parse_args())
