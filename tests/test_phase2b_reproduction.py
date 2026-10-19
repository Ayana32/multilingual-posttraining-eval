"""CPU-only reproduction check: regenerating the Phase 2B scorer-validation
summary from only the committed files (data/phase2b/, report/) must
reproduce report/scorer_validation_summary.txt exactly. No GPU, no
network, no raw results directory -- this is the reproducibility
guarantee docs/reproduce-phase2b.md's Level 1 promises outside
researchers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_scorer_validation_summary_is_reproducible_from_committed_files(tmp_path):
    out_path = tmp_path / "scorer_validation_summary.txt"
    result = subprocess.run(
        [
            sys.executable, "scripts/analyze_scorer_validation.py",
            "--data-dir", "data/phase2b",
            "--report-dir", "report",
            "--output", str(out_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    regenerated = out_path.read_text(encoding="utf-8")
    committed = (REPO_ROOT / "report" / "scorer_validation_summary.txt").read_text(encoding="utf-8")
    assert regenerated == committed
