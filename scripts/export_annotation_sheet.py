#!/usr/bin/env python3
"""Export a BLIND manual-annotation sheet of harmful-prompt responses in one
language (no scorer columns, to avoid anchoring). Fill in human_refusal,
understood_request, response_language, notes; then run
scripts/summarize_annotations.py. See mpe.analysis.comprehension.

Example (Korean SFT/DPO/RLVR, with English prompts shown for reference):
    python scripts/export_annotation_sheet.py \\
        --run-id <sft_ko_run> --run-id <dpo_ko_run> --run-id <rlvr_ko_run> \\
        --reference-run-id <sft_en_run> \\
        --output results/annotation/ko_harmful_blind.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.analysis.comprehension import export_annotation_sheet  # noqa: E402
from mpe.storage.store import ResultStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", action="append", dest="run_ids", required=True)
    parser.add_argument("--reference-run-id", action="append", dest="reference_run_ids", default=None)
    parser.add_argument("--target-language", default="ko")
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--output", default=Path("results/annotation/harmful_blind.csv"), type=Path)
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    path = export_annotation_sheet(
        store,
        args.run_ids,
        args.output,
        target_language=args.target_language,
        reference_run_ids=args.reference_run_ids,
    )
    n = sum(1 for _ in open(path, encoding="utf-8-sig")) - 1
    print(f"{n} rows written to {path}")
    print("Labels: human_refusal=refusal|partial|compliance, understood_request=yes|no|unclear, "
          "response_language=ko|en|mixed|other")


if __name__ == "__main__":
    main()
