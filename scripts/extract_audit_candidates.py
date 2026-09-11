#!/usr/bin/env python3
"""Extract scorer-audit candidates from stored results into a CSV review
artifact for manual inspection. Deterministic: no LLM judge, no adjudication
UI, no database -- just ResultStore in, CSV out. See
mpe.analysis.audit.extract_audit_candidates for the selection rules.

Example (audit every run currently in results/):
    python scripts/extract_audit_candidates.py

Example (audit a specific set of runs, e.g. one Phase 2B trajectory):
    python scripts/extract_audit_candidates.py \
        --run-id phase2b_sft_ko_polyguard_abc123 \
        --run-id phase2b_dpo_ko_polyguard_def456 \
        --run-id phase2b_rlvr_ko_polyguard_ghi789 \
        --output results/audit/phase2b_ko_trajectory.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.analysis.audit import extract_audit_candidates, write_audit_candidates_csv  # noqa: E402
from mpe.storage.store import ResultStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        default=None,
        help="Run id to include (repeatable). Omit to audit every run in --results-dir.",
    )
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--output", default=Path("results/audit_candidates.csv"), type=Path)
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    candidates = extract_audit_candidates(store, run_ids=args.run_ids)

    if not candidates:
        print("No audit candidates found -- nothing written.")
        return

    path = write_audit_candidates_csv(candidates, args.output)
    reason_counts: dict[str, int] = {}
    for c in candidates:
        for reason in c.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"{len(candidates)} audit candidates written to {path}")
    print("Reason breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
