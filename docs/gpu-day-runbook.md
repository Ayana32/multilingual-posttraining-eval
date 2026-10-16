# GPU-Day Runbook — Length Ablation → Phase 2B

Status: written and reviewed before any pod is launched. Copy-paste
procedure for the day the 1024-token length ablation (preregistered in
`docs/phase2a-length-ablation.md`) is actually run, followed — gated on
its outcome — by the Phase 2B 200-item EN/KO run. No RunPod started, no
model run, as of this writing.

Do not change `max_new_tokens` on the day (section 6, and
`docs/phase2a-length-ablation.md` section 5) — if 1024 fails a gate, stop
and document; don't silently try a bigger number mid-session.

---

## 1. Pods

**3 pods, one per stage** (SFT, DPO, RLVR) — each pod runs both languages
for its stage sequentially, not one pod per (stage, language). All three:

- **RunPod A40 48GB**, Community Cloud (same GPU type as the 200-token
  pilot runs — see `docs/phase2a-pilot.md` section 4 for the VRAM
  calculation: ~17GB realistic peak for generator + scorer, so 48GB is
  comfortable headroom, not a tight fit).
- **Same base image** across all three pods (record the exact image
  name/tag used when launching — it goes in the env log, step 2).
- **Same git commit** checked out on all three — confirm
  `git rev-parse HEAD` matches across all three pods before any pod
  starts generating; a length-ablation G2 prefix mismatch is a different
  finding if the three pods aren't even running the same code.
- **Disk**: large enough for one resident 7B checkpoint (~15GB bf16
  weights) plus PolyGuard-Qwen-Smol (~1GB bf16) plus the HF cache
  overhead for both — 60GB+ disk recommended per pod so re-downloads or
  cache growth across the pilot + Phase 2B runs on the same pod don't
  hit disk pressure mid-run.

No pod is launched by running any command in this runbook — launching is
a manual RunPod console/CLI action outside this document's scope, done
only after the pods/pricing are explicitly approved on the day.

---

## 2. Per-pod setup

Run on **each of the 3 pods** before any generation:

```bash
# 1. Confirm the GPU is visible:
nvidia-smi

# 2. Clone at the agreed commit:
git clone https://github.com/Ayana32/multilingual-posttraining-eval.git
cd multilingual-posttraining-eval
git rev-parse HEAD   # confirm this matches the other two pods

# 3. Install:
pip install -e '.[hf]'

# 4. CUDA check -- must print True before going further:
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 5. Hugging Face auth (read-only scope is sufficient -- every repo used
#    here is public/ungated):
hf auth login
hf auth whoami
```

**Write an environment log** (one per pod, so a prefix mismatch under G2
can be checked against what each pod was actually running):

```bash
STAGE=sft   # or dpo / rlvr -- set per pod
mkdir -p results
{
  echo "stage: $STAGE"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit: $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  python -c "import torch; print('cuda_version:', torch.version.cuda)"
  python -c "import torch; print('torch:', torch.__version__)"
  python -c "import transformers; print('transformers:', transformers.__version__)"
  python -c "import accelerate; print('accelerate:', accelerate.__version__)"
} > results/env_${STAGE}.txt
cat results/env_${STAGE}.txt
```

**This log matters for G2.** `docs/phase2a-length-ablation.md`'s G2
(prefix fidelity) explicitly does not auto-fail a 200-vs-1024 text
divergence, because greedy decoding is not guaranteed bit-identical
across different sequence lengths, batch compositions, or
kernel/hardware paths even at `temperature=0`. If G2 reports real
divergences (not boundary-only), the first thing to check is whether the
GPU/driver/CUDA/torch/transformers/accelerate versions in
`results/env_<stage>.txt` differ from whatever ran the original
200-token pilot — a version or hardware difference is a plausible,
non-alarming explanation; an identical environment with real divergences
is a more concerning signal and should be investigated further before
trusting the 1024-token data.

---

## 3. Per-pod pilot run

On each pod, run that stage's **EN config, then KO config**, both at
1024 tokens, with the real evaluator and the real scorer (not mock):

```bash
# SFT pod:
time python scripts/run_experiment.py \
    --experiment-config configs/experiments/pilot_sft_en_polyguard_len1024.yaml \
    --evaluator hf --scorer polyguard
time python scripts/run_experiment.py \
    --experiment-config configs/experiments/pilot_sft_ko_polyguard_len1024.yaml \
    --evaluator hf --scorer polyguard

# DPO pod:
time python scripts/run_experiment.py \
    --experiment-config configs/experiments/pilot_dpo_en_polyguard_len1024.yaml \
    --evaluator hf --scorer polyguard
time python scripts/run_experiment.py \
    --experiment-config configs/experiments/pilot_dpo_ko_polyguard_len1024.yaml \
    --evaluator hf --scorer polyguard

# RLVR pod:
time python scripts/run_experiment.py \
    --experiment-config configs/experiments/pilot_rlvr_en_polyguard_len1024.yaml \
    --evaluator hf --scorer polyguard
time python scripts/run_experiment.py \
    --experiment-config configs/experiments/pilot_rlvr_ko_polyguard_len1024.yaml \
    --evaluator hf --scorer polyguard
```

Note the printed `run_id` for each (needed by step 5's pairs file, and
`make_length_pairs.py` discovers these automatically by
`experiment_name` so the exact run_id string doesn't need to be
transcribed by hand). Record the `real` wall-clock time printed by
`time` for each of the 6 runs (EN and KO per stage) — this is the actual
per-run cost at 1024 tokens, useful for estimating the Phase 2B runs
(200 items, same per-item cost) in step 6.

Watch for the stop conditions in section 7 while these run — don't let a
systematic failure on item 1 burn through all 50 items before noticing.

---

## 4. Backup immediately after each run

**Before doing anything else** — before starting the next config, before
moving on to step 5 — copy that run's directory off the pod:

```bash
# From your Mac, not the pod:
scp -r <pod-user>@<pod-host>:/workspace/multilingual-posttraining-eval/results/<run_id> ./results/

# Verify the line count matches on both ends before trusting the copy:
wc -l ./results/<run_id>/records.jsonl                                     # on the Mac
ssh <pod-user>@<pod-host> "wc -l /workspace/multilingual-posttraining-eval/results/<run_id>/records.jsonl"   # on the pod
```

Both must report the same count (50 for a pilot run). An unverified copy
is not a preserved result (`docs/phase2a-pilot.md` section 8.7) — do this
for each of the 6 pilot runs before moving to the gate step, and again
for each of the 6 Phase 2B runs in step 6 if the gate says go.

---

## 5. Gate on the Mac

Once all 6 len1024 pilot runs are backed up and line-count-verified
locally, **keep the pods running** (do not terminate yet — the decision
in step 6 determines whether they do more work) and run the gate
computation locally:

```bash
# Build the pairs file automatically (discovers the 6 len1024 runs by
# experiment_name, pairs them with the known 200-token run IDs from
# docs/phase2a-length-ablation.md section 1 -- fails loudly on anything
# ambiguous or known-bad, see scripts/make_length_pairs.py):
python scripts/make_length_pairs.py --out results/length_ablation_pairs.json

# Compute the full gate report:
python scripts/compare_length_ablation.py \
    --pairs-file results/length_ablation_pairs.json \
    --gate-verdict-out results/length_ablation_gate_verdict.json \
    --report-out results/length_ablation_report.txt \
    --review-csv-out results/length_ablation_review.csv \
    --review-key-csv-out results/length_ablation_review_key.csv
```

Read `results/length_ablation_report.txt` in full (G0 through G3,
transitions, stage outcomes), not just the final verdict line — the
verdict only encodes G0/G1a; G2/G3 need a human read per
`docs/phase2a-length-ablation.md` section 5.

---

## 6. Decision

- **Go**: `overall_go: true` in the gate verdict **and** G2/G3 show
  nothing systematic on manual read (isolated/boundary-only G2
  divergences are fine; a systematic jump in empty completions, parse
  failures, or real divergences across a whole run is not — see
  `docs/phase2a-length-ablation.md` sections G2/G3). Proceed: each pod
  runs its stage's two `phase2b_*` configs, **EN then KO**, same backup
  step (section 4) after each:

  ```bash
  # SFT pod:
  time python scripts/run_experiment.py \
      --experiment-config configs/experiments/phase2b_sft_en_polyguard.yaml \
      --evaluator hf --scorer polyguard
  time python scripts/run_experiment.py \
      --experiment-config configs/experiments/phase2b_sft_ko_polyguard.yaml \
      --evaluator hf --scorer polyguard
  # (DPO / RLVR pods: substitute phase2b_dpo_*/phase2b_rlvr_* respectively)
  ```

  **Do not change `max_new_tokens`** for these runs or anywhere else
  today — the configs already have it fixed at 1024 and that is a
  decision this ablation exists to validate, not re-open on the day.

- **No-go**: `overall_go: false`, or G2/G3 show a systematic problem.
  **Stop all pods** (do not run any `phase2b_*` config) and document the
  failure per `docs/phase2a-length-ablation.md` section 5 — which gate
  failed, the actual numbers, and (per that section) do not silently
  retry at a larger token budget; any further length change needs its
  own diagnosis and its own recorded decision.

---

## 7. Stop conditions

Copied from `docs/phase2a-pilot.md` section 8.9 — stop and do not
proceed to the next step if:

- **OOM**: `torch.cuda.OutOfMemoryError` (or a CUDA OOM surfaced through
  the per-item `except Exception` handler as an `error:` `finish_reason`)
  on more than an isolated item or two. A handful of isolated errors are
  captured gracefully and don't lose other results; a systematic OOM
  across most items means the GPU choice needs revisiting before
  spending more, not powering through.
- **Template mismatch**: if the raw-continuation Base path (not expected
  to be hit here — SFT/DPO/RLVR all have real chat templates) somehow
  produces output containing chat-template special tokens
  (`<|im_start|>` etc.), the no-chat-template detection may be wrong for
  this specific checkpoint — stop and re-verify `supports_chat_template()`
  against the actual loaded tokenizer's `.chat_template` attribute
  directly, don't assume a prior finding still holds at generation time.
- **Scorer parser mismatch**: if PolyGuard's raw output doesn't match
  the expected `Harmful request:` / `Response refusal:` / `Harmful
  response:` line format on manual read of the first few outputs, stop
  — do not proceed past that point on a parser that isn't reading what
  the scorer is actually saying.
- **Scorer failure**: if `PolyGuardScorer.score()` itself raises (as
  opposed to producing unparseable text) for a meaningful fraction of
  items, stop — this is a different failure mode from a parsing
  mismatch and points at a scorer-loading or scorer-generation problem,
  not a regex problem.

---

## 8. Shutdown checklist

Before terminating any pod:

- [ ] All 6 len1024 pilot run directories backed up to the Mac and
      line-count-verified on both ends (section 4).
- [ ] If the gate was a go: all 6 `phase2b_*` run directories also
      backed up and line-count-verified (section 4).
- [ ] All 3 `results/env_<stage>.txt` environment logs copied to the Mac.
- [ ] `results/length_ablation_gate_verdict.json`,
      `length_ablation_report.txt`, `length_ablation_review.csv`, and
      `length_ablation_review_key.csv` (section 5) are present locally.
- [ ] Only then: terminate all 3 pods.

An unverified backup is not a preserved result — do not terminate a pod
on the assumption that a copy "probably went through."
