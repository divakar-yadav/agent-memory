#!/usr/bin/env python3
"""REGEN retrieval engine, runnable inside an agent sandbox.

The agent calls this instead of walking the memory tree:

    python3 regen_engine.py "When did Caroline go to the support group?"

It applies the full gate -- signal fidelity sigma with a stage-widening match
window, multiplied by the retention prior gamma^k -- and prints only the
memories that clear the threshold, ranked, with provenance. One bash call
replaces a directory traversal, and the agent may re-run with a reworded
query, which is what makes reconstruction iterative.

DEVIATION FROM THE MEASURED PIPELINE: the context term of sigma uses BM25
over the corpus rather than embedding cosine. Embedding an arbitrary,
agent-reworded query at call time would require shipping an API credential
into the sandbox, which we decline to do. BM25 is deterministic, needs no
key, and handles rephrasing -- but it is lexical, so sigma here is a weaker
signal than in the local pipeline. The lifecycle (stage, gate, tiers, depth)
is unchanged.

Self-contained: numpy only, data sharded alongside this file.
"""
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# calibrated parameters (paper Sec. 3.2); THETA set from the score
# distribution so the gate actually prunes -- see calibrate_theta.py
LAM, GAMMA, K = 1.2, 0.62, 8
DELTA0, ETA = 1.0, 0.35
THETA = float(os.environ.get("REGEN_THETA", "0.38"))
ALPHA, BETA = 1.8, 0.45
BM25_K1, BM25_B = 1.5, 0.75


def toks(s):
    return [w for w in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(w) > 2]


def load():
    recs = []
    for f in sorted(glob.glob(os.path.join(HERE, "shard_*.json"))):
        with open(f) as fh:
            recs.extend(json.load(fh))
    for r in recs:
        r["_tok"] = toks(f"{r['t']} {r['tg']} {r['tp']}")
    return recs


def bm25(recs, query):
    """Lexical similarity in [0,1], standing in for embedding cosine."""
    N = len(recs)
    df = Counter()
    for r in recs:
        for w in set(r["_tok"]):
            df[w] += 1
    avgdl = sum(len(r["_tok"]) for r in recs) / max(1, N)
    q = toks(query)
    scores = np.zeros(N, dtype=np.float32)
    for i, r in enumerate(recs):
        tf = Counter(r["_tok"])
        dl = len(r["_tok"])
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * (tf[w] * (BM25_K1 + 1)) / (
                tf[w] + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
        scores[i] = s
    m = float(scores.max())
    return scores / m if m > 0 else scores          # normalise to [0,1]


def main():
    if len(sys.argv) < 2:
        print('usage: regen_engine.py "<question>" [max_results]')
        return 1
    query, topn = sys.argv[1], (int(sys.argv[2]) if len(sys.argv) > 2 else 12)

    recs = load()
    sim = bm25(recs, query)

    rows = []
    for i, r in enumerate(recs):
        k = int(r["k"])
        d2 = max(0.0, 1.0 - float(sim[i]))          # context distance
        dk = DELTA0 * (1.0 + ETA * k)               # stage-widening window
        sigma = math.exp(-d2 / (2.0 * dk * dk))
        score = sigma * (GAMMA ** k)                # gate: sigma * retention prior
        if score >= THETA:
            rows.append((score, sigma, k, r))
    rows.sort(key=lambda x: -x[0])

    kbar = float(np.median([x[2] for x in rows])) if rows else 2.0
    sbar = float(np.median([x[1] for x in rows])) if rows else 0.3
    depth = max(1, math.ceil(ALPHA * max(kbar, 0.5) / (sbar + BETA)))

    print(f"# regen engine | {len(rows)}/{len(recs)} cleared gate "
          f"(theta={THETA}) | predicted depth D={depth}")
    if not rows:
        print("# nothing cleared the gate -- rephrase with more specific terms")
        return 0
    for score, sigma, k, r in rows[:topn]:
        tier = int(r["l"])
        body = r["t"] if tier == 0 else (r["g"] or r["t"][:80])
        prov = "recalled" if tier == 0 else f"inferred(tier={tier})"
        age = "" if r["d"] is None else f" [t+{int(r['d'])}d]"
        print(f"- ({r['tg']}){age} <{prov}> sigma={sigma:.2f} k={k}\n    {body}")
    if len(rows) > topn:
        print(f"# {len(rows)-topn} more below the cut; rephrase to narrow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
