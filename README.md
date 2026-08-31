# Multilingual Post-Training Safety Evaluation Framework

A reproducible framework — and an initial pilot study built on it — for evaluating whether safety-relevant refusal behavior learned through LLM post-training (SFT → DPO → RLVR) transfers consistently from English to another language, using the same underlying prompts in both languages.

## Motivation

Post-training safety evaluation is overwhelmingly conducted in English, even though the resulting models are deployed multilingually. It is not obvious that refusal behavior calibrated during English-centric post-training generalizes to other languages with the same reliability. This project builds the infrastructure to test that question directly and reproducibly, rather than assuming it.

## Research Question

Does harmful-prompt refusal behavior, measured on the exact same parallel prompts, hold consistently between English and Korean across a model's post-training trajectory (SFT → DPO → RLVR) — or does it diverge?

## Architecture

The pipeline is config-driven and stage-separated: an `ExperimentConfig` (Pydantic, YAML-loadable) specifies a checkpoint lineage, stages, languages, benchmarks, and generation settings; `ExperimentRunner` sequences dataset loading → generation → parsing → scoring → storage for every (stage, benchmark, language) combination the config asks for; results are persisted as append-only JSONL via `ResultStore`, keyed by run ID.

```
ExperimentConfig ─▶ CheckpointRegistry ─▶ DatasetLoader ─▶ Evaluator
                                                              │
                                                              ▼
                                          Scorer  ◀─  ParsedOutput
                                            │
                                            ▼
                                        ResultStore  ─▶  mpe.analysis
```

Every component behind this pipeline is swappable independently: a `MockEvaluator` exercises the full pipeline deterministically with no model or network access; `HFTransformersEvaluator` performs real generation via Hugging Face `transformers`. Nothing downstream of `ResultStore` re-runs generation or scoring — analysis only ever reads what was already stored.

## Design Principles

- **Deterministic sampling.** Benchmark item sampling is seeded (`deterministic_sample()`), so the same seed/limit always reproduces the same item set.
- **Parallel-by-construction cross-lingual sampling.** PolyGuardPrompts' English and Korean subsets share a common `id` per translated prompt; sampling on that shared ID (independent of which language is loaded) is what keeps English and Korean runs aligned to the same underlying prompts.
- **Base is excluded from causal trajectory comparisons.** The Base checkpoint has no chat template and is evaluated via raw prompt continuation — a categorically different generation protocol from SFT/DPO/RLVR's shared chat template. It is retained only as a descriptive reference point, never as stage 0 of a trajectory delta.
- **No fabricated results.** `MockEvaluator` output is used exclusively for pipeline/unit testing and is never presented as, or used to compute, a research finding. All reported pilot numbers come from `HFTransformersEvaluator` against real checkpoints.
- **Scorer output is kept raw and auditable.** The authoritative scorer's literal text output is stored alongside its parsed verdict, and a parse failure is recorded as `parse_ok=False` rather than guessed.
- **Analysis is deterministic Python, not model-generated.** Summaries, comparisons, and comparability checks are plain, tested functions over stored records — no LLM is involved in computing or interpreting results.

## Repository Structure

```
.github/workflows/ci.yml            CI: install + run the deterministic test suite
configs/benchmarks/registry.yaml    Benchmark registry
configs/experiments/                Per-run experiment configs (this pilot's SFT/DPO/RLVR x EN/KO configs)
configs/models/olmo3_lineages.yaml  Verified OLMo 3 checkpoint lineage/stage registry
docs/phase0-findings.md             Research design and dataset/checkpoint verification
docs/phase1-notes.md                Phase 1 implementation notes
docs/phase1-review.md               Phase 1 review
docs/phase2a-pilot.md               Full pilot methodology, evidence, and results
scripts/inspect_scorer_outputs.py   Manual scorer-output inspection helper
scripts/run_experiment.py           CLI entry point
src/mpe/analysis/                   Deterministic run summary/comparison utilities
src/mpe/checkpoints/                Checkpoint/lineage schema + registry
src/mpe/config/                     ExperimentConfig
src/mpe/datasets/                   Benchmark loaders (PolyGuardPrompts, XSTest, Belebele)
src/mpe/evaluators/                 Mock and Hugging Face evaluators
src/mpe/metrics/                    Pure scoring/aggregation functions
src/mpe/parsing/                    Raw-completion output parsing
src/mpe/runner/                     ExperimentRunner orchestration
src/mpe/scorers/                    Authoritative Tier-2 scorer (PolyGuard)
src/mpe/storage/                    ResultRecord schema + JSONL ResultStore
tests/                              Unit + integration tests
```

## Implemented Components

| Component | Status |
|---|---|
| `PolyGuardPromptsLoader` (EN/KO, parallel sampling) | Implemented, used in the completed pilot |
| `XSTestLoader` | Implemented, not yet run as part of the completed pilot |
| Belebele loader | Implemented, not yet run as part of the completed pilot |
| `MockEvaluator` | Implemented; testing only, never a source of reported results |
| `HFTransformersEvaluator` | Implemented, used for all real pilot generation |
| `PolyGuardScorer` (Tier-2 authoritative scorer) | Implemented, used in the completed pilot |
| Tier-1 rule-based output parsing | Implemented |
| `ResultStore` (JSONL, append-only) | Implemented |
| `mpe.analysis` (deterministic summary/comparison) | Implemented, validated against real stored pilot results |
| CI (GitHub Actions) | Implemented |

## Deterministic Run Analysis

`mpe.analysis` provides two pure functions over stored `ResultRecord`s — no new scoring or generation logic, only composition of existing metrics:

- **`get_run_summary(store, run_id)`** — parseable rate, language-match rate, scorer parse rate, overall refusal rate, and harmful-prompt vs. unharmful-prompt refusal rate, each derived from the record's own stored `expected_label` and `scorer_response_refusal` fields.
- **`compare_runs(store, run_a, run_b)`** — checks sample comparability before computing anything: identical `parallel_item_id` sets, no duplicate `(language, parallel_item_id)` pairs, and matching record counts (so a multi-language run can't be mistaken as aligned with a single-language run sharing the same bare IDs). A separate `comparable_trajectory` flag additionally requires the same lineage, the same single language on both sides, the same benchmark, and both stages within the comparable trajectory set (excluding Base). Metric deltas are still returned whenever the sample aligns — even across languages or when Base is involved — but `comparable_trajectory` and an explicit warning make clear when a comparison is not a clean stage-over-stage effect.

## Pilot Experiment

| | |
|---|---|
| Model | OLMo 3 7B (Instruct lineage) |
| Stages | SFT, DPO, RLVR |
| Languages | English, Korean |
| Benchmark | PolyGuardPrompts |
| Sample | 50 parallel prompts per run (28 harmful / 22 unharmful), identical `parallel_item_id` set across every corresponding EN/KO and stage run |
| Evaluator | `HFTransformersEvaluator` — real model inference, not `MockEvaluator` |
| Scorer | `PolyGuardScorer` (ToxicityPrompts/PolyGuard-Qwen-Smol) |
| Compute | RunPod, 1x NVIDIA A40 |

## Pilot Results

Harmful-prompt refusal rate, same 50 parallel items throughout:

| Stage | English | Korean | EN / KO ratio |
|---|---|---|---|
| SFT | 82.1% | 32.1% | ~2.6x |
| DPO | 71.4% | 28.6% | ~2.5x |
| RLVR | 71.4% | 25.0% | ~2.9x |

Across all three post-training stages, harmful-prompt refusal was observed to be roughly **2.5–2.9x lower in Korean than in English on the identical prompts**, and the gap did not close as post-training progressed.

## Interpretation and Limitations

This is a **pilot-scale observation, not a statistically validated finding**:

- n = 50 prompts per run (28 harmful), a single seed, no repeated sampling.
- One model family (OLMo 3 7B), two languages.
- No bootstrap confidence intervals or formal significance testing has been performed.
- Comparisons across released checkpoints (SFT/DPO/RLVR) are **observational**, not causal — differences are associated with checkpoint stage, not attributed to the training procedure that produced each stage.
- Base is reported only as a descriptive reference (raw-continuation protocol), never as part of the SFT→DPO→RLVR trajectory.
- Manual inspection of scorer output surfaced apparent false positives (e.g. purely educational/definitional responses to benign prompts flagged as harmful), observed recurring across multiple stages and languages on at least one item — see `docs/phase2a-pilot.md` for the full inspection record.

## Scorer Reliability

The authoritative scorer (`PolyGuardScorer`, `ToxicityPrompts/PolyGuard-Qwen-Smol`) parses a fixed three-line verdict format (`Harmful request`, `Response refusal`, `Harmful response`) via regex, and only reports a verdict as trustworthy (`parse_ok=True`) if all three labels are found — a partial or malformed match is recorded as a parse failure, never guessed. This parser's expected format was empirically validated against real model output during the pilot (documented in `docs/phase2a-pilot.md`), and scorer parse success was consistently at or near 100% across all real pilot runs. Independently of parse success, manual review found the scorer's *judgment* itself is not fully reliable — occasional apparent false positives were observed by manual inspection, most notably one recurring case that was flagged as harmful across five separate real-model runs despite reading as benign on manual review. Scorer output is therefore treated as a strong signal, not ground truth.

## Development Setup

```bash
# Minimum: run the deterministic test suite (matches CI exactly)
pip install -e ".[dev]" torch

# Real Hugging Face inference (HFTransformersEvaluator, PolyGuardScorer)
pip install -e ".[hf]"

# Both, for developing and running real models from the same environment
pip install -e ".[dev,hf]"
```

`[dev]` installs `pytest` and `mypy` on top of the base dependencies (`pydantic`, `pyyaml`, `requests`, `pandas`, `pyarrow`). `[dev]` alone is **not** sufficient to pass the full test suite: three evaluator tests import `torch` directly, so `torch` must be installed alongside it (as CI does). `[hf]` installs `transformers`, `torch`, and `accelerate` and is required for any real generation via `HFTransformersEvaluator` or `PolyGuardScorer`.

No GPU is required for installation or for running the test suite.

## Running an Experiment

```bash
# Quick, deterministic sanity check -- no model weights, no network after first dataset cache
python scripts/run_experiment.py \
  --experiment-config configs/experiments/smoke_test.yaml \
  --evaluator mock

# Real model inference + authoritative scoring
python scripts/run_experiment.py \
  --experiment-config configs/experiments/pilot_dpo_ko_polyguard.yaml \
  --evaluator hf \
  --scorer polyguard
```

| Flag | Required | Choices | Default |
|---|---|---|---|
| `--experiment-config` | Yes | any YAML path | — |
| `--checkpoint-registry` | No | any YAML path | `configs/models/olmo3_lineages.yaml` |
| `--results-dir` | No | any path | `results` |
| `--evaluator` | No | `mock`, `hf` | `mock` |
| `--scorer` | No | `polyguard` | none (Tier-1 parsing only) |

## Testing

```bash
pytest              # full suite -- 126 tests, requires torch (see Development Setup)
pytest -m "not network"   # CI-equivalent: excludes the 3 tests that need the real
                           # cached PolyGuardPrompts dataset; deterministic and network-free
```

The `network` marker is reserved for tests that pin properties of the real, full PolyGuardPrompts dataset (exact row count, exact null-label rows) that cannot be reproduced against a synthetic fixture; everything else in the suite runs against small in-memory fixtures with no network or model dependency.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request: Ubuntu, Python 3.12, `pip install -e ".[dev]" torch`, then `pytest -m "not network"`. No GPU, no Hugging Face downloads, and no real-model inference happen in CI. The full local suite is 126 tests; CI runs the network-free subset of that same suite and currently passes on the latest commit.

## Cloud GPU Execution

Real-model generation (`--evaluator hf`) is computationally infeasible on CPU for a 7B model at pilot scale in reasonable time and was run on a RunPod Secure Cloud pod (1x NVIDIA A40) instead, using this repository's own CLI and configs directly.

## Real-Run Analysis Validation

`get_run_summary()` and `compare_runs()` were also run directly against real stored pilot results — e.g. the Korean DPO run, and the Korean DPO vs. RLVR comparison — and the computed rates matched the figures derived by hand during the pilot's manual analysis, confirming the layer is correct against genuine model output, not only its own test fixtures.

## Project Status

**Implemented and validated:**
- Dataset and checkpoint verification (Phase 0/1)
- Config-driven evaluation pipeline (loaders, evaluators, parsing, scoring, storage)
- Completed EN/KO x SFT/DPO/RLVR pilot on PolyGuardPrompts (real model inference)
- Deterministic run analysis layer, validated against real pilot output
- CI (deterministic, network-free test suite)

**Planned next steps:**
- Larger-scale evaluation: bigger samples and multiple seeds per stage/language
- Additional languages beyond English and Korean
- Belebele-based competence controls, to separate "doesn't understand the language" from "understands but declines"
- Further scorer reliability auditing beyond the manual spot-checks performed so far
- Statistical uncertainty quantification (confidence intervals, significance testing) once sample size supports it
- Structured reporting of pilot results beyond the current per-run JSONL records
- Durable, versioned storage for run artifacts (currently local JSONL under `results/`)

This project is an active research-engineering pilot, **not a production system**.

## Broader Goal

This pilot exists to establish working, tested, reproducible infrastructure and a first real signal before committing to a larger study. The observed ~2.5–2.9x English/Korean refusal gap, if it held under a properly powered study (larger sample, multiple seeds, significance testing, additional model families and languages), would suggest that safety alignment learned through post-training does not transfer uniformly across languages — a question worth answering rigorously, not assuming.
