#!/usr/bin/env python
"""LoCoMo benchmark runner for REGEN vs. baselines.

Usage:
  python run_benchmark.py --convs 2 --questions 40 --sessions 12
  python run_benchmark.py --aging          # age-stratified recall
  python run_benchmark.py --continual      # continual replay at 1x/4x/16x

All LLM calls are disk-cached, so re-running is free after the first pass.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from regen_mem.agents import ActiveAgent, RAGAgent, RegenAgent
from regen_mem.graph import MemoryGraph, Params
from regen_mem.ingest import build_graph, parse_locomo_date
from regen_mem.llm import LLM, USAGE

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "locomo10.json"
CACHE = ROOT / ".graphs"
RESULTS = ROOT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

JUDGE = """You are grading a question-answering system. Output valid JSON only.

Judge whether the PREDICTION is semantically equivalent to the GOLD answer.
Accept paraphrase, different surface form, extra harmless detail, and
equivalent date formats. Reject wrong entities, wrong dates, wrong counts,
and refusals.

QUESTION: {q}
GOLD: {gold}
PREDICTION: {pred}

{{"correct": true|false}}
"""


def judge(llm: LLM, q: str, gold: str, pred: str) -> bool:
    if not pred.strip():
        return False
    try:
        d = llm.complete_json(
            JUDGE.format(q=q, gold=gold, pred=pred), phase="judge", max_tokens=1500
        )
        return bool(d.get("correct"))
    except Exception:  # noqa: BLE001
        return False


def f1(pred: str, gold: str) -> float:
    pt, gt = pred.lower().split(), gold.lower().split()
    if not pt or not gt:
        return 0.0
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    n = sum(min(pt.count(w), gt.count(w)) for w in common)
    p, r = n / len(pt), n / len(gt)
    return 2 * p * r / (p + r) if p + r else 0.0


def evidence_recall(pred_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return float("nan")
    got = set(pred_ids)
    return sum(1 for g in gold_ids if g in got) / len(gold_ids)


def get_graph(sample: dict, llm: LLM, params: Params, sessions: int | None) -> tuple[MemoryGraph, datetime]:
    sid = sample.get("sample_id", "s")
    cf = CACHE / f"{sid}_{sessions}_{llm.model}.json"
    conv = sample["conversation"]
    dates = [parse_locomo_date(v) for k, v in conv.items() if k.endswith("_date_time")]
    dates = [d for d in dates if d]
    epoch = min(dates) if dates else datetime(2023, 1, 1)

    if cf.exists():
        return MemoryGraph.load(cf, params), epoch
    t0 = time.time()
    g = build_graph(sample, llm, params, max_sessions=sessions)
    print(f"    built {len(g.contents)} nodes in {time.time()-t0:.0f}s")
    g.save(cf)
    return g, epoch


def run(args) -> None:
    random.seed(0)
    llm = LLM(args.model)
    judge_llm = LLM(args.judge_model)
    params = Params()
    samples = json.loads(DATA.read_text())[: args.convs]

    print(f"model={args.model}  judge={args.judge_model}")
    print(f"tau (irrecoverability stage) = {params.tau}\n")

    rows: list[dict] = []
    for si, sample in enumerate(samples):
        print(f"[conv {si+1}/{len(samples)}] {sample.get('sample_id')}")
        g, epoch = get_graph(sample, llm, params, args.sessions or None)

        # Questions whose evidence lies inside the ingested session window.
        kept_dia = {d for c in g.contents.values() for d in c.dia_ids}
        qa = [
            q for q in sample["qa"]
            if q.get("category") != 5
            and q.get("evidence")
            and all(e in kept_dia for e in q["evidence"])
        ]
        random.shuffle(qa)
        qa = qa[: args.questions]
        print(f"    {len(g.contents)} nodes, {len(qa)} answerable questions")
        if not qa:
            continue

        # "Now" = one year after the last session, so decay has bitten.
        last_t = max((c.t_days or 0) for c in g.contents.values())
        now = last_t + args.age_days
        g.apply_decay(now)
        if args.replay:
            g.replay()
        print(f"    decay@{now}d -> {g.stats()}")

        for name, mk in [
            ("RAG", lambda gg: RAGAgent(gg, llm)),
            ("Active (MRAgent-style)", lambda gg: ActiveAgent(gg, llm)),
            ("REGEN", lambda gg: RegenAgent(gg, llm, epoch)),
        ]:
            if args.only and name != args.only:
                continue
            # Each system gets its own copy so consolidation does not leak.
            gg = MemoryGraph.load(CACHE / f"{sample.get('sample_id','s')}_{args.sessions or None}_{llm.model}.json", params)
            gg.apply_decay(now)
            if args.replay:
                gg.replay()
            agent = mk(gg)

            USAGE.reset()
            t0 = time.time()
            n_ok = 0
            for qi, q in enumerate(qa):
                a = agent.answer(q["question"])
                ok = judge(judge_llm, q["question"], str(q["answer"]), a.text)
                n_ok += ok
                rows.append({
                    "conv": si, "q": qi, "system": name,
                    "category": CATEGORY.get(q.get("category"), "?"),
                    "question": q["question"], "gold": str(q["answer"]),
                    "pred": a.text, "correct": ok,
                    "f1": f1(a.text, str(q["answer"])),
                    "ev_recall": evidence_recall(a.evidence_dia_ids, q.get("evidence", [])),
                    "turns": a.turns, "materialised": a.n_materialised,
                    "provenance": a.provenance, "debug": a.debug,
                })
                print(f"      {name:24s} {qi+1}/{len(qa)} acc={n_ok/(qi+1):.0%}", end="\r")
            u = USAGE.snapshot()
            el = time.time() - t0
            print(f"      {name:24s} acc={n_ok/len(qa):.1%}  "
                  f"tok/q={u['total_tokens']//max(1,len(qa)):,}  {el:.0f}s        ")
            for r in rows[-len(qa):]:
                r["tokens_per_q"] = u["total_tokens"] / max(1, len(qa))
                r["store_chars"] = gg.stats()["store_chars"]
                r["working_set"] = gg.stats()["working_set"]

    out = RESULTS / args.out
    out.write_text(json.dumps(rows, indent=1, default=str))
    summarise(rows)
    print(f"\nwrote {out}")


def summarise(rows: list[dict]) -> None:
    import statistics as st
    systems = sorted({r["system"] for r in rows}, key=lambda s: (s != "RAG", s))
    cats = ["multi-hop", "temporal", "open-domain", "single-hop"]
    print("\n" + "=" * 78)
    print(f"{'System':26s} " + " ".join(f"{c[:9]:>10s}" for c in cats) + f" {'ALL':>8s} {'tok/q':>9s}")
    print("-" * 78)
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        if not rs:
            continue
        cells = []
        for c in cats:
            sub = [r["correct"] for r in rs if r["category"] == c]
            cells.append(f"{100*sum(sub)/len(sub):9.1f}%" if sub else f"{'-':>10s}")
        allacc = 100 * sum(r["correct"] for r in rs) / len(rs)
        tok = st.mean([r.get("tokens_per_q", 0) for r in rs])
        print(f"{s:26s} " + " ".join(cells) + f" {allacc:7.1f}% {tok:9,.0f}")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--convs", type=int, default=2)
    ap.add_argument("--questions", type=int, default=25)
    ap.add_argument("--sessions", type=int, default=0)  # 0 = all
    ap.add_argument("--age-days", type=int, default=0)
    ap.add_argument("--no-replay", dest="replay", action="store_false", default=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="pilot.json")
    run(ap.parse_args())
