# Phase 3 RunPod Runbook

How Phase 3's R_tune gates (G0-G2) were actually run on RunPod (2026-11-06),
and what to expect/do differently next time. This is operational notes, not
part of the pre-registration (`docs/phase3-visual-injection.md`) or its
deviations log (`docs/phase3-deviations.md`).

## Pod

- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU: 1x NVIDIA A40 (Secure Cloud, ~$0.59/hr)
- Repo setup is a **git clone at a named commit**, not an upload or a bare
  checkout -- this is what makes `/root/repo` a real git repository (a
  `.git` directory present) that `harness.git_commit_hash`/`git_is_dirty`
  can read, rather than something the dirty-repo R_eval guard would refuse
  outright for lacking `.git`:
  ```bash
  git clone https://github.com/Ayana32/multilingual-posttraining-eval.git /root/repo
  cd /root/repo
  git fetch --tags
  git checkout <commit>   # e.g. ebd4030, the harness-code commit -- see the
                          # correction note below for what this hash is NOT
  ```
- The image ships torch already installed. Install, in this exact order,
  from inside `/root/repo` (no venv: installing into the image's system
  Python avoids re-downloading torch):
  ```bash
  pip install -e ".[dev,phase3,hf]"
  pip install transformers==4.57.1 hf_transfer
  ```
  **The second line is not optional.** `pyproject.toml`'s `hf` extra pins
  only `transformers>=4.45`, so the first line alone resolves to whatever
  the latest release is -- on both the gates pod and the phrase-selection
  pod this installed **transformers 5.22.x first**, and only the explicit
  second `pip install transformers==4.57.1 hf_transfer` line downgraded it
  to the frozen `4.57.1` (see `phase3/config.yaml` and
  `docs/phase3-commit-b.md`). Skipping the second line means R_eval runs on
  a different serving stack than the one P1 was tuned and gated on -- see
  the pre-flight check in the R_eval chunk plan below, which exists
  specifically to catch this before chunk_01 runs.
- `pip install -e .` creates `src/mpe.egg-info/` inside the repo; this is
  already covered by `.gitignore`'s `*.egg-info/` entry, confirmed locally
  by cloning this repo fresh, running `pip install -e .`, and checking
  `git status --porcelain` stays empty. Combined with keeping nohup logs
  outside the repo (next section), a pod set up this way starts, and stays,
  clean -- `git_is_dirty` returns `False`, so `start_run(receipt_set="eval")`
  is not spuriously refused. `git_is_dirty` does count untracked files (not
  just modified tracked ones) by design; the fix here is keeping the
  workflow from creating any inside the repo, not loosening that check.

## Environment recorded during the gate run

(See `report/phase3_gate_environment.json`; `git_commit`/`git_dirty`/
`pip_freeze` were added to `harness.start_run`'s environment.json *after*
this run, so they aren't present in the original capture. The code version
for this run was commit `ebd4030` (`ebd403095ba0f207a35d99421bdda4d09213ac4e`,
"Pin Qwen3-VL-8B-Instruct revision for Phase 3 gates"), recorded here and in
that file by hand from the session log.)

| | |
|---|---|
| Python | 3.12.3 |
| torch | 2.8.0+cu128 |
| transformers | 4.57.1 |
| CUDA | 12.8 |
| GPU | NVIDIA A40 |

A full `pip freeze` of the pod's environment was **not** captured during this
run (only a handful of package versions were grepped out individually), so
`report/phase3_gate_pip_freeze.txt` does not exist for this run. Future runs
get a full pip freeze automatically, since `harness.start_run` now writes it
into `environment.json` for every run (see `docs/phase3-deviations.md`-adjacent
change in `phase3/harness.py`).

## A dependency gap in the image

The image sets `HF_HUB_ENABLE_HF_TRANSFER=1`, but does not ship the
`hf_transfer` package that env var requires. The first G2 attempt crashed
during the model download step, before any inference happened:

```
ValueError: Fast download using 'hf_transfer' is enabled
(HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available in
your environment. Try `pip install hf_transfer`.
```

Fix: `pip install hf_transfer`, then re-run. This is environment setup, not a
code or design change -- it was not treated as a gate failure.

## SSH access: proxy only, heredoc-driven, no scp/sftp

The pod's SSH proxy (`ssh <pod>-<id>@ssh.runpod.io`) only supports interactive
PTY sessions:

- A non-interactive `ssh host "command"` fails immediately with `Error: Your
  SSH client doesn't support PTY`.
- `scp`/`sftp` fail with `subsystem request failed on channel 0` -- the proxy
  does not implement the SFTP subsystem. Use `runpodctl send`/`receive` for
  real file transfer if the local `runpodctl` CLI is authenticated; otherwise
  (as here), everything has to go through the interactive shell.

What worked: force a PTY (`ssh -tt ...`) and pipe a script via stdin as a
heredoc, ending with `exit`:

```bash
ssh -tt -i ~/.ssh/id_ed25519_runpod <pod>-<id>@ssh.runpod.io <<'EOF'
<commands, one per line>
exit
EOF
```

Long-running steps (model download, test suite, each gate) were launched with
`nohup ... > log 2>&1 & disown`, so they survive the SSH session ending, then
polled by reconnecting and checking the log/process. The log files (e.g.
`/root/run_g2.log`) are written **outside** `/root/repo` entirely (directly
under `/root/`), not into a gitignored folder inside the repo -- either
would keep the repo clean, but staying outside the repo is simplest and is
what this run actually did.

**Pitfall:** because the pty echoes back whatever is piped into it, a polling
command like `grep -c "G2_END" file.log` self-matches on its own echoed
*command text* (which contains the literal string `G2_END`), reporting
"done" immediately and falsely. Poll with a command whose own text does not
contain the marker you're looking for (e.g. `tail -c 300 file.log`, checked
locally for the marker) instead.

To pull result files back (no scp), their contents were `cat`-ed through the
same interactive session with unique `MARKER_BEGIN_x`/`MARKER_END_x` lines
around each file, captured into one local transcript, then parsed back into
separate files locally (stripping ANSI escapes and shell-prompt lines glued
onto the content).

## Gate commands (R_tune only, run in this order)

```bash
cd /root/repo
python3 phase3/gates.py g2
python3 phase3/gates.py g1b
python3 phase3/gates.py g1a
python3 phase3/gates.py g0
```

Each writes/merges its result into `report/phase3_gate_report.json` **on the
pod** (i.e. `/root/repo/report/phase3_gate_report.json`), and prints
`PASS`/`FAIL`. That file only exists on the pod's disk, though -- getting it
into this repo took two more steps, both manual, since scp/sftp don't work
here: the file contents were pulled back over the SSH heredoc approach above
into `results/phase3/gates/` (gitignored, not committed) first, and only
then copied from there into the committed `report/phase3_gate_report.json`
and `report/phase3_gate_artifacts/` (with the on-pod `/root/repo/...` paths
inside the JSON rewritten to be repo-relative). All four gates passed
(2026-11-06).

## R_eval chunk plan (added 2026-11-07, before Commit B's R_eval runs)

`phase3/run.py` has no resume capability, and `harness.write_records` opens
its output file with `"w"` (truncate) and writes everything once at the end
-- a crash partway through a run loses that run's records, and a re-run of
the same command overwrites the file rather than picking up where it left
off (confirmed by reading `phase3/run.py::main` and
`phase3/harness.py::write_records`; neither reads an existing `records.jsonl`
first). So R_eval runs in chunks, each a separate `run.py` invocation with
its own `--out-dir`, small enough that losing one chunk to a crash is cheap
to redo.

**Eval set** (`phase3/data/manifest.jsonl`, `receipt_set == "eval"`): 20
receipts (`eval_01`-`eval_20`), 7 variants each (`clean`, `neutral_en`,
`neutral_ko`, `P1_en`, `P1_ko`, `P2_en`, `P2_ko`). R_eval only runs the
**frozen** phrasing (P1, see `docs/phase3-commit-b.md`) -- `P2_en`/`P2_ko`
are in the manifest (generated for every receipt, for G3 review) but must
be excluded from every command below with an explicit `--variants` list.

Per Section 4.4's run table, scaled to `n_eval_receipts = 20`: 10 agent runs
per receipt (2 `clean` + 4 `neutral` + 4 `inject`[P1] over 2 channels) x 20
receipts = **200 agent runs**; 4 transcription runs per receipt (`neutral_en`,
`neutral_ko`, `P1_en`, `P1_ko` -- images with an extra line) x 20 = **80
transcription runs**.

### Pre-flight (before chunk_01)

The launcher script below (`run_all_chunks.sh`) runs this same check as its
first step and exits without running any chunk if it fails. Running it by
hand too, right after the install steps in the "Pod" section above, just
gives an earlier/faster signal than waiting for the launcher. It must print
`PRE-FLIGHT OK`; if it doesn't, stop and fix the install (most likely: the
`pip install transformers==4.57.1 hf_transfer` line was skipped or
something later re-pulled a different version) before running anything:

```bash
python3 -c "
import torch, transformers
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
assert torch.__version__ == '2.8.0+cu128', f'torch mismatch: {torch.__version__} != 2.8.0+cu128 (frozen record)'
assert transformers.__version__ == '4.57.1', f'transformers mismatch: {transformers.__version__} != 4.57.1 (frozen record)'
print('PRE-FLIGHT OK')
"
```

This exists because both the gates pod and the phrase-selection pod
installed transformers 5.22.x on `pip install -e ".[dev,phase3,hf]"` alone
(the `hf` extra only pins `transformers>=4.45`), and only the second,
explicit pin corrected it -- if a chunk run skipped that step, R_eval would
run on a different serving stack than what P1 was tuned and gated on,
silently.

### Chunks

8 agent chunks of exactly 25 records each (5 receipts x 5 variants x 1
channel), partitioned by receipt-id group and channel so there is no
overlap; 2 transcription chunks of 40 each (10 receipts x 4 payload-bearing
variants), split the same way the agent chunks are, so no single chunk is
much larger than another. Every chunk gets its own `--out-dir`; never reuse
one.

| chunk | receipt_ids | channel | variants | n |
|---|---|---|---|---|
| chunk_01 | eval_01-eval_05 | image | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_02 | eval_01-eval_05 | text | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_03 | eval_06-eval_10 | image | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_04 | eval_06-eval_10 | text | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_05 | eval_11-eval_15 | image | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_06 | eval_11-eval_15 | text | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_07 | eval_16-eval_20 | image | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_08 | eval_16-eval_20 | text | clean,neutral_en,neutral_ko,P1_en,P1_ko | 25 |
| chunk_transcription_01 | eval_01-eval_10 | n/a | neutral_en,neutral_ko,P1_en,P1_ko | 40 |
| chunk_transcription_02 | eval_11-eval_20 | n/a | neutral_en,neutral_ko,P1_en,P1_ko | 40 |

### Launcher: run all chunks in order, log exit codes, continue on failure

A single launcher script runs every chunk in sequence, logs each chunk's
exit code to `/root/logs/chunk_status.log`, and **continues to the next
chunk even if one fails** (so one bad chunk doesn't block the other nine).
Per-chunk stdout/stderr still goes to its own log file under `/root/logs/`,
**outside** the repo, so `git_is_dirty` stays `False` the same way it did
for phrase selection.

```bash
cat > /root/run_all_chunks.sh <<'SCRIPT_EOF'
#!/bin/bash
cd /root/repo
mkdir -p /root/logs
STATUS_LOG=/root/logs/chunk_status.log
: > "$STATUS_LOG"

python3 -c "
import torch, transformers
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
assert torch.__version__ == '2.8.0+cu128', f'torch mismatch: {torch.__version__} != 2.8.0+cu128 (frozen record)'
assert transformers.__version__ == '4.57.1', f'transformers mismatch: {transformers.__version__} != 4.57.1 (frozen record)'
print('PRE-FLIGHT OK')
"
if [ $? -ne 0 ]; then
  echo "PRE-FLIGHT FAILED: not running any chunk" >&2
  exit 1
fi

run_chunk() {
  local name="$1"; shift
  python3 phase3/run.py "$@" > "/root/logs/${name}.log" 2>&1
  echo "${name} exit_code=$?" >> "$STATUS_LOG"
}

run_chunk chunk_01 --receipt-set eval --channels image --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_01,eval_02,eval_03,eval_04,eval_05 --out-dir results/phase3/eval/chunk_01
run_chunk chunk_02 --receipt-set eval --channels text  --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_01,eval_02,eval_03,eval_04,eval_05 --out-dir results/phase3/eval/chunk_02
run_chunk chunk_03 --receipt-set eval --channels image --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_06,eval_07,eval_08,eval_09,eval_10 --out-dir results/phase3/eval/chunk_03
run_chunk chunk_04 --receipt-set eval --channels text  --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_06,eval_07,eval_08,eval_09,eval_10 --out-dir results/phase3/eval/chunk_04
run_chunk chunk_05 --receipt-set eval --channels image --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_11,eval_12,eval_13,eval_14,eval_15 --out-dir results/phase3/eval/chunk_05
run_chunk chunk_06 --receipt-set eval --channels text  --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_11,eval_12,eval_13,eval_14,eval_15 --out-dir results/phase3/eval/chunk_06
run_chunk chunk_07 --receipt-set eval --channels image --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_16,eval_17,eval_18,eval_19,eval_20 --out-dir results/phase3/eval/chunk_07
run_chunk chunk_08 --receipt-set eval --channels text  --variants clean,neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_16,eval_17,eval_18,eval_19,eval_20 --out-dir results/phase3/eval/chunk_08

run_chunk chunk_transcription_01 --receipt-set eval --variants neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_01,eval_02,eval_03,eval_04,eval_05,eval_06,eval_07,eval_08,eval_09,eval_10 --out-dir results/phase3/eval/chunk_transcription_01 --transcription
run_chunk chunk_transcription_02 --receipt-set eval --variants neutral_en,neutral_ko,P1_en,P1_ko --receipt-ids eval_11,eval_12,eval_13,eval_14,eval_15,eval_16,eval_17,eval_18,eval_19,eval_20 --out-dir results/phase3/eval/chunk_transcription_02 --transcription

cat "$STATUS_LOG"
SCRIPT_EOF
chmod +x /root/run_all_chunks.sh
mkdir -p /root/logs
nohup bash /root/run_all_chunks.sh > /root/logs/run_all_chunks.log 2>&1 &
disown
```

`mkdir -p /root/logs` must run before the `nohup` line: the outer `>
/root/logs/run_all_chunks.log` redirect needs that directory to already
exist, and the script only creates it *inside itself* (too late for the
redirect). On the first R_eval launch attempt (2026-11-07), this line was
missing and the launch failed immediately with no chunk ever running; it
was added by hand before the (successful) relaunch.

After it finishes, check `/root/logs/chunk_status.log` for any
`exit_code != 0`. **Retry only those chunks**, individually, into a
**fresh** `--out-dir` (e.g. `chunk_03_retry`, never reuse `chunk_03`), and
note in the merge step which directory (original or retry) is the one
actually used -- the failed partial directory is left in place for the
record, just excluded from the merge.

Each chunk writes its own `environment.json` (via `harness.start_run`), so
every check below runs per chunk, not just once -- catching a mid-sequence
edit, an accidental dirty tree, or a version drift on any single chunk.

### Merge check (run once, after all 10 chunks, using each chunk's final --out-dir -- original or retry)

1. `sum(len(records.jsonl) for chunk_01..chunk_08) == 200`.
2. `len(chunk_transcription_01/transcription_records.jsonl) + len(chunk_transcription_02/transcription_records.jsonl) == 80`.
3. No duplicate `(receipt_id, variant, channel)` triple across chunk_01-08's
   combined agent records (each should appear exactly once).
4. Every chunk's `environment.json` matches the frozen record
   (`docs/phase3-commit-b.md`) on:
   - `git_commit` == the Commit B commit hash (check with
     `git rev-parse phase3-prereg-B`),
   - `git_dirty` == `false`,
   - `transformers_version` == `"4.57.1"`,
   - `torch_version` == `"2.8.0+cu128"`,
   - `gpu_name` == `"NVIDIA A40"`.

## R_eval run log (2026-11-07)

- **Pod:** `qptjbw6pg6db19` (A40, Secure Cloud, 100 GB disk -- bumped up
  from the gates/phrase-selection pods' 60 GB for this run), terminated
  after results were pulled back and verified.
- **Started:** 2026-11-07 13:35:55 UTC (second launch attempt, after
  adding `mkdir -p /root/logs`; the first attempt's exact failure time
  wasn't logged separately, but it never got past the redirect error --
  see the correction above the launcher script).
- **Ended:** finished between 13:41 and 14:36 UTC (at 14:36 I checked
  `chunk_status.log` in the RunPod web terminal and all 10 chunks showed
  `exit_code=0`).
- **Result:** all 10 chunks `exit_code=0`; 200/200 agent records
  `outcome_status == "valid"` (no `parse_failure`, no `harness_error`);
  0/200 agent records and 0/80 transcription records ever hit
  `max_new_tokens`; merge check (record counts, no duplicate
  `(receipt_id, variant, channel)`, every chunk's `environment.json`
  matching the frozen record) passed in full. See
  `report/phase3_eval/SHA256SUMS` and the per-chunk files there for the
  raw records; M1-M3/TASK/CO have not been scored yet.

## Correction (added 2026-11-06, after commit 835a833)

Commit 835a833's message says "Qwen3-VL-8B-Instruct @ ebd4030". `ebd4030` is
the **harness code** commit (`phase3/config.yaml`'s revision was pinned in
that commit), not the **model's** revision. The model (Qwen3-VL-8B-Instruct)
was pinned to `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (see
`phase3/config.yaml`'s `model_revision`). The commit message is left as-is
(no amend, no force push); this note is the correction.
