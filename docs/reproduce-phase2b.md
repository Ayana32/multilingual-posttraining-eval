# Reproducing the Phase 2B results

Two levels. Level 1 needs no GPU and takes minutes; it reproduces every
number in `docs/phase2b-results.md` from the committed, redacted data.
Level 2 needs a GPU and real model access; it regenerates the raw
completions themselves, which cannot be published (see Data Availability
in the README and in `docs/phase2b-results.md`).

## Level 1: CPU, minutes

```bash
git clone https://github.com/Ayana32/multilingual-posttraining-eval.git
cd multilingual-posttraining-eval
pip install -e ".[dev]" torch   # torch is only needed for the unrelated evaluator tests

python scripts/analyze_phase2b.py --data-dir data/phase2b
python scripts/analyze_phase2b_followup.py --data-dir data/phase2b
python scripts/analyze_scorer_validation.py --data-dir data/phase2b --report-dir report
```

The third command writes `report/scorer_validation_summary.txt`. Compare it against the
committed copy:

```bash
diff report/scorer_validation_summary.txt <(git show HEAD:report/scorer_validation_summary.txt)
```

The bootstrap and Jeffreys-prior draws are seeded (see the `random.Random(seed)` calls in
`scripts/analyze_scorer_validation.py`), so this diff should be empty. The same check also runs
as a test:

```bash
pytest tests/test_phase2b_reproduction.py -v
```

This is also exercised in CI, the same `pytest -m "not network"` command the rest of the suite
runs under; a failure means the committed `report/scorer_validation_summary.txt` is out of date
with the committed `data/phase2b/` or `report/human_labels_*.csv`.

The output of `analyze_phase2b.py` and `analyze_phase2b_followup.py` has no single committed
snapshot file to diff against (unlike the scorer-validation summary), but every number either
script prints should match the corresponding number quoted in `docs/phase2b-results.md`
sections 3 and 4.

None of these three scripts read a prompt or a completion. They read only
`data/phase2b/<run_id>/records_redacted.jsonl` (redacted Phase 2B records: item IDs, labels,
scorer verdicts, generation config, and the precomputed `completion_sha256`,
`completion_char_len`, `dup_fraction`, `max_5gram_count`, `is_collapse`) and
`report/human_labels_{en,ko}.csv` (the blind human annotations, joined by item ID). See Data
Availability below for why the raw completions themselves are not in this list.

## Level 2: GPU, real completions

This regenerates the actual model completions, which only matters if you want to check the
completions themselves (e.g. against `completion_sha256`) rather than just the aggregate
numbers, which Level 1 already fully covers.

```bash
# On a GPU pod, same commit:
python scripts/run_experiment.py --experiment-config configs/experiments/phase2b_dpo_en_polyguard.yaml --evaluator hf --scorer polyguard
python scripts/run_experiment.py --experiment-config configs/experiments/phase2b_dpo_ko_polyguard.yaml --evaluator hf --scorer polyguard
python scripts/run_experiment.py --experiment-config configs/experiments/phase2b_rlvr_en_polyguard.yaml --evaluator hf --scorer polyguard
python scripts/run_experiment.py --experiment-config configs/experiments/phase2b_rlvr_ko_polyguard.yaml --evaluator hf --scorer polyguard

# Then, back where you have both the new raw results and this repo:
python scripts/build_redacted_records.py /path/to/new/results --out-dir /tmp/new_phase2b
```

Compare `completion_sha256` between `/tmp/new_phase2b/<run_id>/records_redacted.jsonl` and the
committed `data/phase2b/<run_id>/records_redacted.jsonl` for the same `item_id`. A mismatch on
some items does not by itself indicate an error.

**Hardware actually used for the committed Phase 2B runs**: the DPO runs (EN and KO) ran on an
NVIDIA A40. The RLVR runs were rerun on an NVIDIA L40 (not the A40 used for the earlier,
50-item pilot and gate stage on this same checkpoint). Greedy decoding (`temperature=0.0`) is
not guaranteed to produce byte-identical completions across different GPU architectures, driver
versions, or library versions, even for the same checkpoint and the same prompt; see
`docs/phase2a-length-ablation.md` section G2 for the same caveat raised at the pilot stage. A
hash mismatch on some items under a different GPU or a different `torch`/`transformers` version
is expected, not necessarily a sign that something is wrong, and should be investigated the way
G2 prefix divergences are investigated there, not treated as an automatic failure.

**Exact package versions**: `results/env_dpo.txt` and `results/env_rlvr.txt` record the
environment captured during this project's earlier 50-item pilot and length-ablation gate stage
on the DPO and RLVR pods (2026-10-16), not the 200-item Phase 2B generation itself, which ran
two days later (`created_at` timestamps in the records are 2026-10-18) and did not have its own
environment log captured. The pilot-stage logs (`torch==2.8.0+cu128`, `transformers==5.20.1`,
`accelerate==1.15.0`, CUDA 12.8) are the closest available record of the toolchain in use around
that time, but were not re-captured for the Phase 2B runs specifically, and the RLVR runs are
also known to have moved to different hardware (L40) in between; treat them as approximate
context, not as the exact versions the committed Phase 2B completions were generated with.
Future runs should save `nvidia-smi` and `pip freeze` output into each results folder, so this
gap does not recur.
