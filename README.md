# Graded forgetting for capacity-bounded agent memory

Code and results for *Demotion Beats Deletion: Graded Forgetting for
Capacity-Bounded Agent Memory*.

Deployed agent memory meets capacity limits by deleting — Anthropic's Memory
Stores cap at 10,000 entries and instruct operators to remove stale ones, Mem0
prunes via LLM-issued `DELETE`, ChatGPT's background pass rewrites in place.
This repository asks whether deletion is necessary, by implementing a lifecycle
that demotes and compresses instead, and measuring what survives.

## Headline result

Under a byte budget, on questions whose supporting evidence the budget
destroyed (833-day timeline, 1,115 memories, budget = 35% of full size, n=40):

| policy | accuracy | tokens/q | state of the evidence |
|---|---|---|---|
| full (uncapped) | 75.0% | 3,117 | intact — ceiling |
| delete-oldest | **2.5%** | 2,806 | removed outright |
| graded forgetting | **60.0%** | 33,423 | compressed to gist |

## Results that cut the other way

These are in the repo because they are part of the finding, not despite it.

**It does not improve ordinary retrieval.** Without capacity pressure it ranks
last or near-last in every setting measured:

| setting | flat retrieval | active reconstruction | graded |
|---|---|---|---|
| LoCoMo n=90, gpt-5.6-luna | **53.3** | 47.8 | 38.9 |
| LoCoMo n=90, claude-sonnet-4.5 | **57.8** | 48.9 | 34.4 |
| Memory Stores (flat / staged tree / query engine) | **58.3** | — | 58.3 / 50.0 |
| Mem0 (third-party) | **66.7** | — | — |

**There is a crossover the paper does not yet report.** `run_budget_sweep.py`
adds `delete-redundant` (evict near-duplicates first), a much stronger baseline
than `delete-oldest`, and sweeps the budget:

| budget | full | delete-oldest | delete-redundant | graded |
|---|---|---|---|---|
| 25% | 84.0 | 4.0 | **28.0** | 16.0 |
| 35% | 72.0 | 0.0 | 24.0 | **60.0** |
| 50% (n=8) | 100.0 | 0.0 | 50.0 | **87.5** |

At a 25% budget graded forgetting **loses** to redundancy-aware deletion, because
every memory is crushed to tag-and-topic and a gist stops being worth more than
nothing. The honest claim is a band, not a uniform win. See
`results/budget_sweep.json`.

## What else this turned up

- **Evidence redundancy dominates eviction experiments.** Distillation writes
  each dialogue turn into a median of 8 memories; *no* question depends on a
  single memory. Uniform random eviction of 65% of the store therefore destroys
  a question's evidence with probability 0.65⁸ ≈ 3%. Any protocol that samples
  questions and buckets them by age measures nothing.
- **Date encoding beat every retrieval mechanism.** Storing timestamps as
  metadata and rendering them as day-offsets left 1 of 367 memories with an
  absolute date in its text (Mem0: 46%). Instructing the extractor to resolve
  relative references lifted coverage to 52–59% and temporal accuracy from
  0.0% to 37.5–50.0%. One line of prompt moved the metric further than all the
  retrieval work combined.
- **Mem0's reconciler never fired.** 381 ingestions produced 381 `ADD`s and
  zero `UPDATE`/`DELETE`, because the corpus contains almost no contradictions.
  Its consolidation machinery is real but workload-dependent.

## Layout

```
regen_mem/           the lifecycle: signal vectors, staged decay, tiers, agents
  signals.py         sigma with the stage-widening match window
  graph.py           Cue-Tag-Content store, decay, consolidation, gate
  ingest.py          LLM distillation into memories + signal vectors
  agents.py          flat retrieval, active reconstruction, graded forgetting
run_benchmark.py         main LoCoMo benchmark (3 systems, 1 shared store)
run_controls.py          attribution: closed-book / wrong-memory / real-memory
run_eviction.py          capacity budget with age-stratified questions
run_evicted_recall.py    the headline protocol: conditions on evicted evidence
run_budget_sweep.py      budget curve + delete-redundant baseline
run_memory_stores_ab.py  A/B on real Anthropic Memory Stores
run_mem0.py              third-party Mem0 baseline
run_experiments.py       continual replay, ageing, recovery past tau
sandbox_engine/          retrieval engine runnable inside an agent sandbox
results/                 every run reported in the paper, as JSON
```

## Reproducing

```bash
python3.13 -m venv .venv && .venv/bin/pip install openai anthropic numpy mem0ai chromadb
./scripts/fetch_data.sh                      # LoCoMo, from the authors' repo
export OPENAI_API_KEY=...                    # embeddings (both backbones)
export ANTHROPIC_API_KEY=...                 # answering + judging

.venv/bin/python run_benchmark.py --model claude-sonnet-4-5 --convs 3 --questions 30
.venv/bin/python run_evicted_recall.py -n 40
.venv/bin/python run_budget_sweep.py --budgets 0.25,0.35,0.50 -n 25
```

All LLM calls are disk-cached in `.llm_cache/` (gitignored), so re-runs are
free after the first pass. Memory graphs land in `.graphs/` (gitignored, ~21MB)
and are rebuilt automatically if absent. Construction dominates cost: roughly
19 distillation calls per conversation.

## Caveats worth knowing before citing anything here

- **n is small.** Most comparisons are n=12–40. One question is 2.5–8.3 points.
  Three repeats of one configuration returned 66.7% and a fourth 58.3%; treat
  gaps under two questions as unresolved.
- **`delete-oldest` is a weak baseline.** Recency-ordered eviction is the
  easiest thing to beat. `delete-redundant` in the sweep is the fair comparison.
- **Token cost is 12x.** The reconstruction loop re-serialises the whole
  accumulated evidence set every turn. An implementation artefact, but unfixed.
- **LoCoMo is LLM-generated dialogue**, which is likely why redundancy is so
  high. The eviction analysis may not transfer to real interaction logs.
- **Single corpus, single budget** for the headline number.

## License

Code: MIT. LoCoMo is distributed by its original authors and is fetched, not
vendored.
