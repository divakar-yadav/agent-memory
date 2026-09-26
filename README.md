# Graded forgetting for capacity-bounded agent memory

Code and results for *Demotion Beats Deletion: Graded Forgetting for
Capacity-Bounded Agent Memory*.

Deployed agent memory meets capacity limits by deleting - Anthropic's Memory
Stores cap at 10,000 entries and instruct operators to remove stale ones, Mem0
prunes via LLM-issued `DELETE`, ChatGPT's background pass rewrites in place.
This repository implements a lifecycle that demotes and compresses instead, and
measures what survives when the store will not fit.

## Headline result

Same byte budget, same questions. One policy erases memories to fit; the other
sheds detail and erases nothing.

| policy | store | memories | accuracy | how the budget is met |
|---|---|---|---|---|
| full (uncapped) | 132.1 KB | 1,115 | 75.0% | no bound - ceiling |
| delete-oldest | **46.2 KB** | 397 | 2.5% | 718 memories erased |
| **graded forgetting** | **46.2 KB** | **1,115** | **60.0%** | 0 erased, detail shed |

In an identical 46.2 KB footprint, graded forgetting retains **2.8x as many
memories** and answers **60.0%** of questions about the material the budget
cost, against **2.5%** for deletion - recovering 80% of the uncapped ceiling.

## Against the strongest baseline

Mem0 has no capacity policy: across 381 ingestions its reconciler issued 381
`ADD` operations and zero `UPDATE` or `DELETE`, so the store grows linearly and
overflows any budget. Held to the same cap it must delete 279 of 381 memories.

| system | recent 0-60d | mid 61-120d | old 121d+ | all |
|---|---|---|---|---|
| Mem0, evicted to fit (102/381 kept) | 60.0% | 6.2% | 25.0% | 26.7% |
| **graded forgetting** (321/321 kept) | **80.0%** | **50.0%** | **100.0%** | **66.7%** |

The ordering inverts precisely when the budget binds.

## Two findings from building it

- **Evidence redundancy dominates eviction experiments.** Distillation writes
  each dialogue turn into a median of 8 memories; *no* question depends on a
  single memory. Uniform random eviction of 65% of the store therefore destroys
  a question's evidence with probability 0.65^8 = 3%. Any protocol that samples
  questions and buckets them by age is insensitive by construction - it must
  condition on evidence actually having been erased.
- **Date encoding beat every retrieval mechanism.** Storing timestamps as
  metadata and rendering them as day-offsets left 1 of 367 memories with an
  absolute date in its text (Mem0: 46%). Instructing the extractor to resolve
  relative references lifted coverage to 52-59% and temporal accuracy to
  37.5-50.0%.

## Layout

```
regen_mem/           the lifecycle
  signals.py         sigma with the stage-widening match window
  graph.py           Cue-Tag-Content store, staged decay, tiers, admission gate
  ingest.py          LLM distillation into memories + signal vectors
  agents.py          retrieval agents over a shared store
run_evicted_recall.py    the headline protocol: conditions on evicted evidence
run_mem0_eviction.py     Mem0 under the same capacity bound
run_eviction.py          capacity budgets and eviction policies
run_budget_sweep.py      budget curve, incl. a redundancy-aware deletion baseline
run_benchmark.py         LoCoMo harness (shared store across systems)
run_memory_stores_ab.py  A/B on real Anthropic Memory Stores
run_mem0.py              Mem0 baseline
run_controls.py          attribution controls
run_experiments.py       continual replay, ageing, recovery past tau
sandbox_engine/          retrieval engine runnable inside an agent sandbox
results/                 raw JSON for every run
```

## Reproducing

```bash
python3.13 -m venv .venv && .venv/bin/pip install openai anthropic numpy mem0ai chromadb
./scripts/fetch_data.sh                      # LoCoMo, from the authors' repo
export OPENAI_API_KEY=...                    # embeddings (both backbones)
export ANTHROPIC_API_KEY=...                 # answering + judging

.venv/bin/python run_evicted_recall.py -n 40          # headline result
.venv/bin/python run_mem0_eviction.py --budget 0.35 -n 30
```

All LLM calls are disk-cached in `.llm_cache/` (gitignored), so re-runs are free
after the first pass. Memory graphs land in `.graphs/` (gitignored, ~21 MB) and
rebuild automatically if absent. Construction dominates cost: roughly 19
distillation calls per conversation.

## Scope and scale

Results are reported on LoCoMo with n = 30-40 questions per condition, one
corpus, and a 35% budget for the headline number. `results/` contains the raw
output of every run in the study, including exploratory and ablation runs beyond
those reported above.

## License

Code: MIT. LoCoMo is distributed by its original authors and is fetched, not
vendored.
