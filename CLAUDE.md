# Project context for Claude Code

Read this first. It tells you what exists, what is settled, and what is open.

## Two papers, two folders

| folder | venue | status |
|---|---|---|
| `iclr2027/` | ICLR 2027 | **submitted — do not change the claims** |
| `mlsys2027/` | MLSys 2027, deadline **30 Oct 2026 20:00 UTC** | active work |

`regen_mem/` is a shared library both papers import. `data/` and `scripts/` are
shared. The ICLR scripts carry a two-line `sys.path` shim so they resolve
`regen_mem` from the parent directory.

## What the ICLR paper established (settled; reuse, don't relitigate)

Under a fixed byte budget, shedding *detail* beats deleting *items*:

| policy | store | memories | accuracy on evicted-past questions |
|---|---|---|---|
| full (uncapped) | 132.1 KB | 1,115 | 75.0% |
| delete-oldest | 46.2 KB | 397 | 2.5% |
| graded forgetting | 46.2 KB | 1,115 | 60.0% |

Mem0 under the same bound reaches 26.7% vs 66.7%, and has no capacity policy of
its own (381 ingestions produced 381 `ADD`, zero `UPDATE`/`DELETE`).

Facts worth knowing before you design an experiment:

- **Evidence is duplicated across a median of 8 memories.** No question depends
  on a single memory. Uniform random eviction of 65% destroys a question's
  evidence only ~3% of the time, so any protocol that samples questions and
  buckets by age is insensitive by construction.
- **Absolute dates must be written into memory text.** Storing a timestamp as
  metadata and rendering a day-offset left 1 of 367 memories with a usable
  date; fixing the extraction prompt moved temporal accuracy more than every
  retrieval mechanism combined.
- **n is small throughout** (12–40 per condition). One question is 2.5–8.3
  points. Treat gaps under two questions as unresolved.

## What the MLSys paper is about

Graded forgetting currently compacts with a hand-written tier ladder. This paper
asks what the ladder *should* be, framed as lossy source coding with side
information at the decoder (Wyner–Ziv): the encoder compacts without knowing the
future query; the decoder sees cues the encoder did not. The design consequence
is that one should discard what the cues will make recoverable and spend bits on
what they will not — which is not what a summariser does.

**The ceiling bounds everything.** Measured on the reference corpus:

| after compaction | share | implication |
|---|---|---|
| answer survives in the gist | 46% | decoder not needed |
| lost but latent in neighbours | 7% | **addressable** |
| unrecoverable | 48% | decoder must abstain |

Those are *question-level*. At the *memory level* the generated dataset splits
43.8 / 23.0 / 33.2. Both are real and measure different things — do not conflate
them in the paper. End-task headroom is the 7%; training signal is the 23%.

**Therefore the headline metric is abstention, not reconstruction fidelity.** A
decoder that always emits text scores well on BLEU and is useless, because a
third of the corpus is unrecoverable. Never report reconstruction BLEU alone.

## Running things

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r mlsys2027/requirements.txt
./scripts/fetch_data.sh            # LoCoMo, from the authors' repo
export OPENAI_API_KEY=...          # embeddings
export ANTHROPIC_API_KEY=...       # answering + judging
```

**Memory graphs are gitignored and must be rebuilt** (~20 min/conversation,
costs API credits):

```bash
cd iclr2027 && ../.venv/bin/python run_benchmark.py \
    --model claude-sonnet-4-5 --convs 3 --questions 30
```

Then the MLSys pipeline:

```bash
cd mlsys2027/data_gen && python build_pairs.py --graphs ../../.graphs --out pairs.jsonl
cd ../model && accelerate launch train_decoder.py --data ../data_gen/pairs.jsonl \
    --base Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --bs 8
```

All LLM calls are disk-cached in `.llm_cache/`, so re-runs are free after the
first pass.

## Open problems

1. **Encoder is unimplemented.** `train_decoder.py` exists; the learned encoder
   (what to discard, given cues arrive later) does not. It is the stronger
   contribution — it improves the 46% survival figure directly, which is worth
   more than the 7% the decoder can chase.
2. **Token cost.** The reconstruction loop re-serialises the whole accumulated
   evidence set each turn: 33k tokens/query vs 2.8k for deletion. An
   implementation artefact, unfixed, and MLSys reviewers will care.
3. **Single corpus.** Everything is LoCoMo, which is LLM-generated and unusually
   redundant. Whether the 8× redundancy holds on real logs is untested and
   determines whether any of this transfers.
4. **Scale the dataset.** 6,492 triples from 7 graphs. LongMemEval, MSC, or
   non-dialogue corpora (commit histories, issue threads) would test transfer.

## House rules

- Never commit `.env`, API keys, or `.llm_cache/`.
- `.gitignore` does **not** support trailing comments — put them on their own
  line, or the pattern silently matches nothing.
- This environment's `grep -r` skips dotfiles; scan per-file before pushing.
- The ICLR paper is anonymous for double-blind review. Do not add author names,
  affiliations, or links to it.
