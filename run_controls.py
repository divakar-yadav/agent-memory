#!/usr/bin/env python
"""Attribution controls: is the model using memory, or priors and inference?

Evidence recall cannot answer this. A system can retrieve the gold turn and
ignore it, or answer correctly from a semantic node that carries no gold id.
The question is settled by ablating the memory itself and seeing what survives.

  closed-book   question only, no evidence at all
                -> the floor from parametric knowledge, guessing and any
                   answer leakage in the question wording
  wrong-memory  evidence from a DIFFERENT conversation, same size
                -> if the score holds up, the evidence was never load-bearing
  real-memory   top-20 from the correct conversation (the RAG condition)

Grounded use of memory is the gap between real-memory and closed-book.
Anything closed-book already gets was never a memory result.
"""
from __future__ import annotations

import argparse, json, random, statistics as st
from pathlib import Path

import numpy as np

from regen_mem.agents import ANSWER_RULES, _render
from regen_mem.graph import MemoryGraph, Params
from regen_mem.llm import LLM, USAGE
from run_benchmark import CATEGORY, judge

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".graphs"
RESULTS = ROOT / "results"


def topk(g: MemoryGraph, llm: LLM, q: str, k: int = 20) -> list[str]:
    cids = list(g.contents)
    E = np.asarray(llm.embed([g.contents[c].text for c in cids], phase="construction_embed"))
    E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    qe = np.asarray(llm.embed([q], phase="retrieval_embed")[0])
    qe /= max(float(np.linalg.norm(qe)), 1e-9)
    return [cids[i] for i in np.argsort(-(E @ qe))[:k]]


def main(a):
    llm, jl, P = LLM(a.model), LLM(a.judge_model), Params()
    samples = json.loads((ROOT / "data" / "locomo10.json").read_text())

    graphs, qas = {}, []
    for s in samples[: a.convs]:
        sid = s["sample_id"]
        g = MemoryGraph.load(CACHE / f"{sid}_None.json", P)
        graphs[sid] = g
        kept = {d for c in g.contents.values() for d in c.dia_ids}
        qa = [q for q in s["qa"] if q.get("category") != 5 and q.get("evidence")
              and all(e in kept for e in q["evidence"])]
        random.Random(0).shuffle(qa)
        qas += [(sid, q) for q in qa[: a.questions]]

    # a donor graph for the wrong-memory condition
    donor_id = [k for k in graphs][-1]
    rows = []

    for cond in ["closed-book", "wrong-memory", "real-memory"]:
        USAGE.reset()
        ok = 0
        for sid, q in qas:
            question = q["question"]
            if cond == "closed-book":
                ev = "(no memory available)"
            else:
                src = graphs[donor_id if cond == "wrong-memory" else sid]
                if cond == "wrong-memory" and sid == donor_id:
                    src = graphs[[k for k in graphs][0]]
                ev = _render(src, topk(src, llm, question), with_tier=False)
            pred = llm.complete(
                f"{ANSWER_RULES}\n\nEVIDENCE:\n{ev}\n\nQUESTION: {question}\n\nAnswer:",
                phase="retrieval", max_tokens=3000).strip()
            good = judge(jl, question, str(q["answer"]), pred)
            ok += good
            rows.append({"condition": cond, "category": CATEGORY.get(q.get("category"), "?"),
                         "question": question, "gold": str(q["answer"]),
                         "pred": pred, "correct": good})
        u = USAGE.snapshot()
        print(f"  {cond:14s} acc={100*ok/len(qas):5.1f}%   tok/q={u['total_tokens']//len(qas):,}")

    print("\n" + "=" * 62)
    print(f"{'condition':16s}" + "".join(f"{c[:9]:>11s}" for c in
          ["multi-hop", "temporal", "open-domain", "single-hop"]) + f"{'ALL':>8s}")
    print("-" * 62)
    for cond in ["closed-book", "wrong-memory", "real-memory"]:
        rs = [r for r in rows if r["condition"] == cond]
        cells = ""
        for c in ["multi-hop", "temporal", "open-domain", "single-hop"]:
            sub = [r["correct"] for r in rs if r["category"] == c]
            cells += f"{100*sum(sub)/len(sub):10.0f}%" if sub else f"{'-':>11s}"
        print(f"{cond:16s}{cells}{100*sum(r['correct'] for r in rs)/len(rs):7.1f}%")
    print("=" * 62)

    cb = 100 * sum(r["correct"] for r in rows if r["condition"] == "closed-book") / len(qas)
    wm = 100 * sum(r["correct"] for r in rows if r["condition"] == "wrong-memory") / len(qas)
    rm = 100 * sum(r["correct"] for r in rows if r["condition"] == "real-memory") / len(qas)
    print(f"\n  attributable to memory  = real - closed-book = {rm-cb:+.1f} points")
    print(f"  attributable to priors  = closed-book         = {cb:.1f} points")
    print(f"  distractor contamination= wrong - closed-book = {wm-cb:+.1f} points")

    (RESULTS / a.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote results/{a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--convs", type=int, default=3)
    ap.add_argument("--questions", type=int, default=30)
    ap.add_argument("--out", default="controls.json")
    main(ap.parse_args())
