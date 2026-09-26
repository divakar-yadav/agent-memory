#!/usr/bin/env python
"""Recall-after-eviction: what survives when the store will not fit.

Every prior benchmark here ran with no capacity pressure, so nothing was ever
forced out and a non-destructive lifecycle had nothing to prove. This one
imposes a byte budget that all systems must meet, and lets each meet it its own
way:

  full          no budget -- the ceiling
  delete-oldest evict whole memories, oldest first, until under budget
  delete-random evict whole memories at random (control for "which" vs "how")
  regen         delete nothing; demote and compress tiers until under budget

The budget is in BYTES, not items. That is the crux: it lets a system trade
fidelity for coverage. An item-count budget would make the comparison
impossible to win by compressing, which would beg the question.

Questions are stratified by the age of their oldest supporting turn, because
the claim under test is specifically about the distant past.

Usage:
  python run_eviction.py --budget 0.35 -n 60
"""
from __future__ import annotations

import argparse, json, random, re, time
from pathlib import Path

import anthropic

from regen_mem.graph import Content, MemoryGraph, Params
from regen_mem.ingest import parse_locomo_date
from regen_mem.llm import LLM, USAGE, _parse_json
from datetime import datetime
from regen_mem.agents import RAGAgent, RegenAgent, ANSWER_RULES, _render

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
CONVS = ["conv-26", "conv-30", "conv-41"]

JUDGE = """Grade this answer. Output valid JSON only.
Accept paraphrase, different surface form, equivalent dates. Reject wrong
entities, wrong dates, wrong counts, and refusals.

QUESTION: {q}
GOLD: {gold}
PREDICTION: {pred}

{{"correct": true|false}}"""


# ---------------------------------------------------------------- timeline
def build_timeline(params: Params):
    """Concatenate the conversations into ONE long history.

    Each conversation keeps its internal day offsets; conversations are laid
    end to end with a gap, so conv-26 is the distant past and conv-41 the
    recent past. Gold evidence ids stay attached to their memories.
    """
    g = MemoryGraph(params)
    qa_all, offset, GAP = [], 0.0, 120.0
    for sid in CONVS:
        sub = MemoryGraph.load(ROOT / ".graphs" / f"{sid}_None_claude-sonnet-4-5.json", params)
        span = max((c.t_days or 0) for c in sub.contents.values())
        sample = next(s for s in json.loads((ROOT / "data" / "locomo10.json").read_text())
                      if s["sample_id"] == sid)
        conv = sample["conversation"]
        keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                      key=lambda k: int(k.split("_")[1]))
        dates = {k: parse_locomo_date(conv.get(f"{k}_date_time", "")) for k in keys}
        ep = min(d for d in dates.values() if d)
        day = {t["dia_id"]: (dates[k] - ep).days + offset
               for k in keys for t in (conv.get(k) or []) if t.get("dia_id") and dates[k]}

        for c in sub.contents.values():
            c2 = Content(cid=f"{sid}:{c.cid}", kind=c.kind, text=c.text, gist=c.gist,
                         tag=c.tag, cues=list(c.cues), signal=c.signal,
                         t_days=(c.t_days or 0) + offset, dia_ids=[f"{sid}:{d}" for d in c.dia_ids],
                         topic=c.topic)
            c2.signal.t_days = c2.t_days
            g.add(c2)

        kept = {f"{sid}:{d}" for d in day}
        for q in sample["qa"]:
            if q.get("category") == 5 or not q.get("evidence"):
                continue
            ev = [f"{sid}:{e}" for e in q["evidence"]]
            if all(e in kept for e in ev):
                ages = [day[e.split(":", 1)[1]] for e in ev if e.split(":", 1)[1] in day]
                qa_all.append({**q, "evidence": ev, "conv": sid,
                               "_t": min(ages) if ages else offset})
        offset += span + GAP
    now = max(c.t_days for c in g.contents.values())
    for q in qa_all:
        q["_age"] = now - q["_t"]
    return g, qa_all, now


# ---------------------------------------------------------------- policies
def store_bytes(g: MemoryGraph) -> int:
    return sum(len(c.visible_text()) for c in g.contents.values())


def apply_policy(g: MemoryGraph, policy: str, budget: int, now: float) -> dict:
    """Bring the store under `budget` bytes. Returns what it cost."""
    g.apply_decay(now)
    if policy == "full":
        return {"kept": len(g.contents), "deleted": 0, "bytes": store_bytes(g)}

    if policy == "regen":
        # compress progressively: raise the tier of the oldest material first,
        # deleting nothing, until the budget is met
        for c in g.contents.values():
            c.tier = 0
        order = sorted(g.contents.values(), key=lambda c: -(now - (c.t_days or 0)))
        i = 0
        while store_bytes(g) > budget and i < len(order) * 3:
            c = order[i % len(order)]
            if c.tier < 3:
                c.tier += 1
            i += 1
        return {"kept": len(g.contents), "deleted": 0, "bytes": store_bytes(g)}

    # destructive policies: drop whole memories
    order = (sorted(g.contents.values(), key=lambda c: (c.t_days or 0))
             if policy == "delete-oldest"
             else sorted(g.contents.values(), key=lambda c: random.Random(7).random()))
    if policy == "delete-random":
        order = list(g.contents.values()); random.Random(7).shuffle(order)
    dropped = 0
    for c in order:
        if store_bytes(g) <= budget:
            break
        del g.contents[c.cid]
        dropped += 1
    for idx in (g.cue_index, g.tag_index, g.topic_index):
        for k in list(idx):
            idx[k] = {c for c in idx[k] if c in g.contents}
    return {"kept": len(g.contents), "deleted": dropped, "bytes": store_bytes(g)}


# ---------------------------------------------------------------- run
def main(a):
    llm = LLM(a.model)
    ac = anthropic.Anthropic()
    params = Params()
    base, qa, now = build_timeline(params)
    full_bytes = store_bytes(base)
    budget = int(full_bytes * a.budget)

    random.Random(0).shuffle(qa)
    qa = qa[: a.n]
    B = [("recent", 0, 200), ("mid", 201, 450), ("old", 451, 10**9)]
    print(f"timeline : {len(base.contents)} memories over {now:.0f} days, "
          f"{full_bytes:,} bytes")
    print(f"budget   : {a.budget:.0%} = {budget:,} bytes")
    print(f"questions: {len(qa)}  " +
          " ".join(f"{n}={sum(1 for q in qa if lo<=q['_age']<=hi)}" for n, lo, hi in B))
    print()

    def judge(q, gold, pred):
        if not pred.strip():
            return False
        r = ac.messages.create(model="claude-haiku-4-5", max_tokens=200,
            messages=[{"role": "user", "content": JUDGE.format(q=q, gold=gold, pred=pred)},
                      {"role": "assistant", "content": "{"}])
        d = _parse_json("{" + "".join(b.text for b in r.content if b.type == "text"))
        return bool(d.get("correct")) if isinstance(d, dict) else False

    rows = []
    for policy in a.policies.split(","):
        g = MemoryGraph(params)
        for c in base.contents.values():
            import copy as _c; g.add(_c.deepcopy(c))
        stat = apply_policy(g, policy, budget, now)
        agent = (RegenAgent(g, llm, datetime(2023, 1, 1), consolidate=False)
                 if policy == "regen" else RAGAgent(g, llm, k=20))
        USAGE.reset(); t0 = time.time(); ok = 0
        for i, q in enumerate(qa):
            try:
                ans = agent.answer(q["question"])
                pred = ans.text
            except Exception as e:
                print(f"    ! q{i}: {type(e).__name__}"); pred = ""
            good = judge(q["question"], str(q["answer"]), pred)
            ok += good
            rows.append({"policy": policy, "q": q["question"], "gold": str(q["answer"]),
                         "pred": pred, "correct": good, "age": q["_age"],
                         "conv": q["conv"], "category": q.get("category")})
            json.dump(rows, open(RESULTS / "eviction.json", "w"), indent=1)
            print(f"  {policy:14s} {i+1}/{len(qa)} acc={ok/(i+1):.0%}", end="\r")
        u = USAGE.snapshot()
        print(f"  {policy:14s} acc={100*ok/len(qa):5.1f}%  kept={stat['kept']:4d} "
              f"deleted={stat['deleted']:4d}  bytes={stat['bytes']:,}  "
              f"tok/q={u['total_tokens']//len(qa):,}   ")

    print("\n" + "=" * 64)
    print(f"{'policy':16s}" + "".join(f"{n:>12s}" for n, _, _ in B) + f"{'ALL':>9s}")
    print("-" * 64)
    for policy in a.policies.split(","):
        rs = [r for r in rows if r["policy"] == policy]
        line = f"{policy:16s}"
        for n, lo, hi in B:
            sub = [r for r in rs if lo <= r["age"] <= hi]
            line += f"{100*sum(r['correct'] for r in sub)/len(sub):11.1f}%" if sub else f"{'-':>12s}"
        line += f"{100*sum(r['correct'] for r in rs)/len(rs):8.1f}%"
        print(line)
    print("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--budget", type=float, default=0.35, help="fraction of full store bytes")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--policies", default="full,delete-oldest,delete-random,regen")
    main(ap.parse_args())
