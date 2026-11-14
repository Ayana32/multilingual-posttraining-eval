# Phase 3b RunPod Runbook

Operational notes for running Phase 3b's R_eval on a pod. Not part of the
pre-registration (`docs/phase3b-replication.md`) or its deviations log
(`docs/phase3b-deviations.md`). Mirrors `docs/phase3-runbook.md`; only the
Phase 3b-specific differences are spelled out.

## Pod and install

Same as Phase 3 (`docs/phase3-runbook.md`): A40 Secure Cloud,
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, a **git clone at a named
commit** (so `/root/repo/.git` exists for the guards), then:

```bash
pip install -e ".[dev,phase3,hf]"
pip install transformers==4.57.1 hf_transfer    # NOT optional (see Phase 3 runbook)
```

Check out a commit that has the `phase3b-prereg` tag in its history (the tag's
commit or a descendant); otherwise `guard_phase3b_eval_run` refuses every
`phase3b_eval` row.

## Pre-flight (before any chunk) -- all must pass

Run from `/root/repo`:

```bash
# 1. Tag ancestry: phase3b-prereg must be an ancestor of HEAD.
git merge-base --is-ancestor phase3b-prereg HEAD && echo "TAG ANCESTRY OK" || { echo "FAIL: HEAD is not at/after phase3b-prereg"; exit 1; }

# 2. Clean tree: the dirty-tree guard refuses phase3b_eval on an uncommitted repo.
test -z "$(git status --porcelain)" && echo "CLEAN TREE OK" || { echo "FAIL: dirty tree"; exit 1; }

# 3. Pinned versions.
python3 -c "
import torch, transformers
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert transformers.__version__ == '4.57.1', transformers.__version__
print('VERSIONS OK')
"

# 4. Font hashes (the three committed fonts, incl. arm B's Nanum Gothic).
python3 -c "
import hashlib, pathlib
want = {
  'NanumGothic-Regular.ttf': '76f45ef4a6bcff344c837c95a7dcc26e017e38b5846d5ae0cdcb5b86be2e2d31',
  'NotoSansKR-Variable.ttf': '194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252',
}
d = pathlib.Path('phase3/fonts')
for name, sha in want.items():
    got = hashlib.sha256((d/name).read_bytes()).hexdigest()
    assert got == sha, f'{name}: {got} != {sha}'
print('FONT HASHES OK')
"

# 5. Image hashes: the committed phase3b/data images reproduce on this pod's
#    Pillow/FreeType, AND all G3(a) committed-file checks still pass.
python3 phase3b/g3_check.py --out /tmp/phase3b_g3a_preflight.json
python3 -c "
import json
d = json.load(open('/tmp/phase3b_g3a_preflight.json'))
assert d['all_cmap_pass'] and d['all_bbox_pass'], 'G3(a) cmap/bbox FAIL'
assert d['all_committed_pixels_match'], d['committed_pixels_failures']
assert d['all_committed_transcript_match'], d['committed_transcript_failures']
print('IMAGE HASHES / G3(a) OK')
"
```

If the pod's Pillow/FreeType differ from the committed render environment
(`phase3b/data/environment.json`: Pillow 11.1.0 / FreeType 2.13.3), the pixel
check is not meaningful and the mismatch is logged in
`docs/phase3b-deviations.md`, same caveat as Phase 3's image-hash tests.

## Gates (G0, G1a, G1b, G2) on the tune receipts

Phase 3b re-uses Phase 3's existing tune receipts and Phase 3's own pass
criteria unchanged (`docs/phase3b-replication.md` Section 10), so the gate
commands are Phase 3's, run in the same order:

```bash
python3 phase3/gates.py g2
python3 phase3/gates.py g1b
python3 phase3/gates.py g1a
python3 phase3/gates.py g0
```

All four must print `PASS` before any Phase 3b evaluation run.

## G0 additional: replay Phase 3's 20 P1_ko image eval runs

Re-run Phase 3's own 20 `P1_ko`/image eval rows into a **gitignored** dir, then
diff against the committed records with `phase3b/g0_compare.py` (Section 10,
"G0, additionally"). This uses Phase 3's own harness/guard and config
(`commit_b_tag: true`, the `phase3-prereg-B` tag in history):

```bash
python3 phase3/run.py --receipt-set eval --channels image --variants P1_ko \
  --receipt-ids eval_01,eval_02,eval_03,eval_04,eval_05,eval_06,eval_07,eval_08,eval_09,eval_10,eval_11,eval_12,eval_13,eval_14,eval_15,eval_16,eval_17,eval_18,eval_19,eval_20 \
  --out-dir results/phase3b/g0_rerun_check

python3 phase3b/g0_compare.py --rerun-dir results/phase3b/g0_rerun_check --out-dir report/phase3b
```

`g0_compare.py` applies the ordered mismatch rule (version diff is the cause to
fix first; otherwise an output diff is a nondeterminism finding, reported with
the re-run copy count next to Phase 3's 16) and writes the committed summary
`report/phase3b/g0_rerun_summary.{json,md}`. It verifies
`report/phase3_eval/SHA256SUMS` before and after. Commit that summary (the raw
re-run records under `results/` are not committed).

## R_eval: run all chunks

```bash
mkdir -p /root/logs
nohup bash phase3b/run_all_chunks.sh > /root/logs/phase3b_run_all.log 2>&1 &
disown
```

The launcher runs every chunk from `phase3b/run.py --list-chunks` (the fixed
chunk plan: 20 agent chunks + 8 transcription chunks = 960 agent + 400
transcription runs), logs each chunk's exit code to
`/root/logs/phase3b_chunk_status.log`, and continues on failure. Per-chunk logs
live under `/root/logs` (outside the repo), so the tree stays clean. Retry only
chunks with a non-zero exit code, each into a **fresh** `--out-dir` (never reuse
one); leave the failed partial dir in place but exclude it from scoring.

Each chunk writes its own `environment.json` via `harness.start_run`, so the
version/`git_dirty` checks below can be made per chunk, same as Phase 3.

## Merge check (after all chunks)

- `sum(agent records across chunks) == 960`; `sum(transcription records) == 400`.
- no duplicate `(arm, font, channel, receipt_id, variant)` across agent chunks.
- every chunk's `environment.json`: `git_dirty == false`, `transformers_version
  == "4.57.1"`, `torch_version == "2.8.0+cu128"`, `gpu_name == "NVIDIA A40"`, and
  `git_commit` at/after `phase3b-prereg`.

## Score

```bash
python3 phase3b/score_eval.py --input-dir results/phase3b/eval --out-dir report/phase3b_results
```

Writes `report/phase3b_results/phase3b_results.{json,md}` with every Section 5-9
measure (H1's one-sided exact McNemar for P1_ko with the Newcombe and bootstrap
intervals; P2_ko descriptively; per-cell copy/conditional/copy_first_call/
worst_case/category and M1-CO with Wilson; false positives; paired EN/KO
discordant; d_i on M2 and on copy; T; arm B per font + the determinism check;
arm C next to arm A). Commit those two files.
