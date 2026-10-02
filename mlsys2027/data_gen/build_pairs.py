#!/usr/bin/env python
"""Generate (compacted, cues, original) training triples from a memory store.

The compressor is its own annotator: every memory compacted to tier l yields a
supervised example whose target is the tier-0 original. No labelling cost.

Each example is also tagged with a recoverability class, which is what makes
the dataset useful for training an *abstaining* decoder:

  survives       the answer-bearing content is still in the compacted form
                 -> decoder should pass through, not elaborate
  latent         dropped here, but present in a neighbouring memory
                 -> decoder SHOULD reconstruct; this is the addressable band
  unrecoverable  dropped and absent from neighbours
                 -> decoder MUST abstain; generating here is confabulation

On the reference corpus these are roughly 46 / 7 / 48 percent. The class
balance is the headline property of the dataset and is reported on build.

Usage:
  python build_pairs.py --graphs ../../.graphs --out pairs.jsonl
  python build_pairs.py --graphs ../../.graphs --out pairs.jsonl --neighbours 40
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from regen_mem.graph import MemoryGraph, Params  # noqa: E402

STOP = {"the", "and", "that", "with", "for", "was", "were", "her", "his", "she",
        "they", "them", "this", "from", "had", "has", "have", "about", "their"}


def toks(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", (s or "").lower())
            if len(w) > 2 and w not in STOP}


def content_tokens(s: str) -> set[str]:
    """Tokens that carry answerable content: entities, numbers, dates.

    Plain token overlap conflates style with substance -- "greeted", "each",
    "after" are lost by any compactor and matter to nobody. We weight toward
    capitalised spans and numerals, which is where gold answers live.
    """
    out = set()
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b|\b\d[\d,:/-]*\b", s or ""):
        out.add(m.group(0).lower().strip(",.:/-"))
    return out


def neighbours(g: MemoryGraph, cid: str, k: int) -> list[str]:
    c = g.contents[cid]
    pool = (g.tag_index.get((c.tag or "").lower().strip(), set())
            | g.topic_index.get((c.topic or "").lower().strip(), set()))
    pool.discard(cid)
    return list(pool)[:k]


def classify(original: str, compacted: str, nbr_texts: list[str]) -> str:
    lost = content_tokens(original) - content_tokens(compacted)
    if not lost:
        return "survives"
    nbr = set()
    for t in nbr_texts:
        nbr |= content_tokens(t)
    return "latent" if len(lost & nbr) / len(lost) >= 0.5 else "unrecoverable"


def build(args) -> None:
    rng = random.Random(args.seed)
    gdir = Path(args.graphs)
    files = sorted(gdir.glob("*_None_*.json"))
    if not files:
        sys.exit(f"no memory graphs under {gdir} -- run the ICLR benchmark first "
                 f"to build them, or point --graphs elsewhere")

    rows, cls = [], Counter()
    for f in files:
        g = MemoryGraph.load(f, Params())
        last = max((c.t_days or 0) for c in g.contents.values())
        g.apply_decay(last)
        for cid, c in g.contents.items():
            original = c.text
            if not original:
                continue
            nbr_ids = neighbours(g, cid, args.neighbours)
            nbr_gists = [g.contents[n].gist or g.contents[n].text[:80] for n in nbr_ids]

            for tier in (1, 2, 3):
                saved, c.tier = c.tier, tier
                compacted = c.visible_text()
                c.tier = saved
                if compacted.strip() == original.strip():
                    continue                       # tier is a no-op here

                k = rng.choice(args.cue_counts)    # vary cue strength (see paper 6)
                cues = {
                    "keywords": c.cues[: rng.randint(2, max(2, len(c.cues)))],
                    "tag": c.tag,
                    "topic": c.topic,
                    "t_days": c.t_days,
                    "neighbour_gists": nbr_gists[:k],
                }
                label = classify(original, compacted, nbr_gists[:k])
                cls[label] += 1
                rows.append({
                    "source": f.stem, "cid": cid, "tier": tier,
                    "compacted": compacted, "cues": cues,
                    "original": original, "recoverability": label,
                    "n_cue_neighbours": k,
                })

    out = Path(args.out)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    tot = sum(cls.values())
    print(f"wrote {len(rows):,} triples from {len(files)} graphs -> {out}")
    print("\nrecoverability distribution (the dataset's headline property):")
    for k in ("survives", "latent", "unrecoverable"):
        n = cls[k]
        note = {"survives": "decoder passes through",
                "latent": "ADDRESSABLE - decoder should reconstruct",
                "unrecoverable": "decoder MUST abstain"}[k]
        print(f"  {k:15s} {n:6,d}  {100*n/tot:5.1f}%   {note}")
    print(f"\n  addressable band = {100*cls['latent']/tot:.1f}% of examples; a decoder")
    print(f"  that emits output on the {100*cls['unrecoverable']/tot:.1f}% unrecoverable")
    print(f"  slice is confabulating by construction.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="../../.graphs")
    ap.add_argument("--out", default="pairs.jsonl")
    ap.add_argument("--neighbours", type=int, default=40)
    ap.add_argument("--cue-counts", type=int, nargs="+", default=[0, 2, 5, 10, 20],
                    help="cue-strength levels sampled per example")
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())
