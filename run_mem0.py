#!/usr/bin/env python
"""Mem0 as a third-party baseline on the same LoCoMo questions.

Every other arm in this study is our own code. Mem0 is a real, published
memory system, so it is the one number here that is not a reimplementation.

Fairness notes:
  * Mem0 does its OWN construction (LLM fact extraction), not our graph. That
    is the honest comparison -- its construction pipeline is part of the
    system under test, unlike our three-way where the graph was held fixed.
  * Same answering backbone (claude-sonnet-4-5), same embedder
    (text-embedding-3-small), same question set, same judge.
  * Answer generation uses our ANSWER_RULES prompt so the output format is
    comparable; only retrieval differs.

Usage:  python run_mem0.py --ingest --run -n 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
STORE = ROOT / ".mem0_store"

ANSWER_RULES = """Answer rules:
- Yes/No questions: answer Yes or No.
- Date/time questions: give the specific date as stated in the evidence.
- Counting questions: give the number.
- Otherwise: give the minimal concrete entity or phrase. Do not write a sentence.
If the evidence does not contain the answer, give your best inference from it.
Never answer "unknown" if any evidence bears on the question."""

JUDGE = """Grade this answer. Output valid JSON only.

Accept paraphrase, different surface form, equivalent dates. Reject wrong
entities, wrong dates, wrong counts, and refusals.

QUESTION: {q}
GOLD: {gold}
PREDICTION: {pred}

{{"correct": true|false}}"""

USER_ID = "locomo_conv26"


def _patch_anthropic_temperature():
    """mem0's Anthropic provider passes `temperature`, which anthropic>=1.7
    removed from messages.create (adaptive-thinking models reject it). Strip
    it at the client boundary so we can keep the backbone constant across
    arms instead of switching mem0 to a different vendor for extraction.
    """
    import anthropic
    orig = anthropic.resources.messages.Messages.create
    def create(self, *a, **kw):
        kw.pop("temperature", None)
        kw.pop("top_p", None)
        return orig(self, *a, **kw)
    anthropic.resources.messages.Messages.create = create


def make_memory():
    _patch_anthropic_temperature()
    from mem0 import Memory
    STORE.mkdir(exist_ok=True)
    return Memory.from_config({
        "llm": {"provider": "anthropic",
                "config": {"model": "claude-sonnet-4-5", "max_tokens": 4000}},
        "embedder": {"provider": "openai",
                     "config": {"model": "text-embedding-3-small"}},
        "vector_store": {"provider": "chroma",
                         "config": {"collection_name": "locomo",
                                    "path": str(STORE)}},
    })


def load_sample():
    return next(s for s in json.loads((ROOT / "data" / "locomo10.json").read_text())
                if s["sample_id"] == "conv-26")


def ingest(args):
    import re
    m = make_memory()
    conv = load_sample()["conversation"]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    t0 = time.time()
    for i, sk in enumerate(keys):
        turns = conv.get(sk) or []
        if not turns:
            continue
        # Mem0 consumes chat turns; give it the session verbatim plus its date
        msgs = [{"role": "user" if t.get("speaker") == conv.get("speaker_a") else "assistant",
                 "content": f"[{conv.get(sk + '_date_time','')}] {t.get('speaker')}: {t.get('text','')}"}
                for t in turns]
        m.add(messages=msgs, user_id=USER_ID)
        print(f"  {sk}: {len(turns)} turns ingested ({time.time()-t0:.0f}s)", flush=True)
    total = len(m.get_all(filters={"user_id": USER_ID}, top_k=5000).get("results", []))
    print(f"\nmem0 extracted {total} memories from {len(keys)} sessions "
          f"in {time.time()-t0:.0f}s")


def run(args):
    from regen_mem.llm import _parse_json
    m = make_memory()
    ac = anthropic.Anthropic()
    sample = load_sample()

    # same question set as every other arm
    from regen_mem.graph import MemoryGraph, Params
    g = MemoryGraph.load(ROOT / ".graphs" / "conv-26_None_claude-sonnet-4-5.json", Params())
    kept = {d for x in g.contents.values() for d in x.dia_ids}
    qa = [q for q in sample["qa"] if q.get("category") != 5 and q.get("evidence")
          and all(e in kept for e in q["evidence"])]
    random.Random(0).shuffle(qa)
    qa = qa[: args.n]

    def judge(q, gold, pred):
        if not pred.strip():
            return False
        r = ac.messages.create(model="claude-haiku-4-5", max_tokens=200,
            messages=[{"role": "user", "content": JUDGE.format(q=q, gold=gold, pred=pred)},
                      {"role": "assistant", "content": "{"}])
        d = _parse_json("{" + "".join(b.text for b in r.content if b.type == "text"))
        return bool(d.get("correct")) if isinstance(d, dict) else False

    rows, ok, toks = [], 0, 0
    for i, q in enumerate(qa):
        hits = m.search(query=q["question"], filters={"user_id": USER_ID}, top_k=args.k)
        mems = hits.get("results", []) if isinstance(hits, dict) else hits
        ev = "\n".join(f"- {h.get('memory','')}" for h in mems) or "(nothing retrieved)"
        r = ac.messages.create(model="claude-sonnet-4-5", max_tokens=1000,
            messages=[{"role": "user",
                       "content": f"{ANSWER_RULES}\n\nEVIDENCE:\n{ev}\n\n"
                                  f"QUESTION: {q['question']}\n\nAnswer:"}])
        pred = "".join(b.text for b in r.content if b.type == "text").strip()
        tok = r.usage.input_tokens + r.usage.output_tokens
        good = judge(q["question"], str(q["answer"]), pred)
        ok += good; toks += tok
        rows.append({"condition": "mem0", "q": q["question"], "gold": str(q["answer"]),
                     "pred": pred, "correct": good, "tokens": tok,
                     "retrieved": len(mems), "category": q.get("category")})
        json.dump(rows, open(RESULTS / "mem0.json", "w"), indent=1)
        print(f"  {i+1}/{len(qa)} acc={ok/(i+1):.0%} retrieved={len(mems)} tok={tok}",
              flush=True)
    print(f"\nmem0  acc={100*ok/len(qa):5.1f}%  tok/q={toks//len(qa):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("-k", type=int, default=20)
    a = ap.parse_args()
    if a.ingest: ingest(a)
    if a.run: run(a)
