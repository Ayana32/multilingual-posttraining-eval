#!/bin/bash
# Phase 3b R_eval launcher (docs/phase3b-replication.md Section 9;
# docs/phase3b-runbook.md). Runs every chunk from phase3b/run.py's fixed chunk
# plan in sequence, logs each chunk's exit code to a status log, and CONTINUES
# on failure so one bad chunk does not block the rest. Same pattern as Phase 3's
# run_all_chunks.sh: per-chunk logs live OUTSIDE the repo (under /root/logs), so
# git_is_dirty stays False and the dirty-tree eval guard is not tripped.
#
# Usage (on the pod, from /root/repo, after the pre-flight in the runbook):
#   bash phase3b/run_all_chunks.sh
set -u
cd "$(dirname "$0")/.." || exit 1          # repo root

LOG_DIR=${PHASE3B_LOG_DIR:-/root/logs}
mkdir -p "$LOG_DIR"
STATUS_LOG="$LOG_DIR/phase3b_chunk_status.log"
: > "$STATUS_LOG"

# Pre-flight: pinned versions must match the frozen record before any chunk.
python3 -c "
import torch, transformers
assert torch.__version__ == '2.8.0+cu128', f'torch mismatch: {torch.__version__}'
assert transformers.__version__ == '4.57.1', f'transformers mismatch: {transformers.__version__}'
print('PRE-FLIGHT OK')
" || { echo 'PRE-FLIGHT FAILED: not running any chunk' >&2; exit 1; }

run_chunk() {
  local name="$1"
  python3 phase3b/run.py --chunk "$name" --out-dir "results/phase3b/eval/$name" \
      > "$LOG_DIR/$name.log" 2>&1
  echo "$name exit_code=$?" >> "$STATUS_LOG"
}

for name in $(python3 phase3b/run.py --list-chunks); do
  run_chunk "$name"
done

echo "=== chunk status ==="
cat "$STATUS_LOG"
