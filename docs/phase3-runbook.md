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
- The image ships torch already installed; `pip install -e ".[dev,phase3,hf]"`
  plus a pinned `transformers==4.57.1` (see `phase3/config.yaml`) is layered on
  top of it, run from inside `/root/repo`. No venv: installing into the
  image's system Python avoids re-downloading torch.
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

## Correction (added 2026-11-06, after commit 835a833)

Commit 835a833's message says "Qwen3-VL-8B-Instruct @ ebd4030". `ebd4030` is
the **harness code** commit (`phase3/config.yaml`'s revision was pinned in
that commit), not the **model's** revision. The model (Qwen3-VL-8B-Instruct)
was pinned to `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (see
`phase3/config.yaml`'s `model_revision`). The commit message is left as-is
(no amend, no force push); this note is the correction.
