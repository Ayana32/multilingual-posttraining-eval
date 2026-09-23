#!/usr/bin/env python3
"""Summarize a filled annotation sheet: human refusal counts, whether
non-refusals understood the request, response language, and scorer/human
agreement (joined from stored records). Blank rows count as unannotated.

Example:
    python scripts/summarize_annotations.py results/annotation/ko_harmful_blind.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.analysis.comprehension import summarize_annotations  # noqa: E402
from mpe.storage.store import ResultStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--json", action="store_true", help="Print the full summary as JSON.")
    args = parser.parse_args()

    summary = summarize_annotations(ResultStore(args.results_dir), args.sheet)
    if args.json:
        print(summary.model_dump_json(indent=2))
        return
    for r in summary.runs:
        kappa = "n/a" if r.cohen_kappa is None else f"{r.cohen_kappa:.2f}"
        agree = "n/a" if r.agreement_rate is None else f"{r.agreement_rate:.0%}"
        print(f"[{r.stage}] {r.run_id}: annotated {r.n_annotated}/{r.n_rows}")
        print(f"  human verdicts:                {r.human_refusal_counts}")
        print(f"  non-refusals understood?       {r.understood_counts_among_non_refusals}")
        print(f"  response language:             {r.response_language_counts}")
        print(f"  scorer vs human (n={r.n_agreement_pairs}):      agreement {agree}, kappa {kappa}")
        print(f"  scorer missed a human refusal: {r.scorer_compliance_human_refusal}")
        print(f"  scorer refusal, human complied: {r.scorer_refusal_human_compliance}")


if __name__ == "__main__":
    main()
