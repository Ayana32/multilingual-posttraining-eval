# Phase 3 RunPod Runbook

How Phase 3's R_tune gates (G0-G2) were actually run on RunPod (2026-11-06),
and what to expect/do differently next time. This is operational notes, not
part of the pre-registration (`docs/phase3-visual-injection.md`) or its
deviations log (`docs/phase3-deviations.md`).

## Pod

- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU: 1x NVIDIA A40 (Secure Cloud, ~$0.59/hr)
- The image ships torch already installed; `pip install -e ".[dev,phase3,hf]"`
  plus a pinned `transformers==4.57.1` (see `phase3/config.yaml`) is layered on
  top of it. No venv: installing into the image's system Python avoids
  re-downloading torch.

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
polled by reconnecting and checking the log/process.

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

Each writes/merges its result into `report/phase3_gate_report.json` and
prints `PASS`/`FAIL`. All four passed (2026-11-06); see
`report/phase3_gate_report.json` and `report/phase3_gate_artifacts/` for the
G2 rendered prompts.
