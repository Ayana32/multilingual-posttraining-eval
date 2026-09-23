#!/usr/bin/env python3
"""Exact McNemar test between two aligned runs (paired by parallel_item_id).

Allowed comparisons match mpe.analysis.bootstrap: same stage across
languages (e.g. SFT-EN vs SFT-KO) or same language across stages (e.g.
SFT-KO vs RLVR-KO). Pass --pair repeatedly to test several at once.

Example (the three cross-language pilot comparisons):
    python scripts/mcnemar_paired.py \\
        --pair <sft_en_run> <sft_ko_run> \\
        --pair <dpo_en_run> <dpo_ko_run> \\
        --pair <rlvr_en_run> <rlvr_ko_run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.analysis.mcnemar import mcnemar_paired  # noqa: E402
from mpe.storage.store import ResultStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--condition", choices=["harmful", "unharmful"], default="harmful")
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    header = f"{'run_a':<40} {'run_b':<40} {'n':>3} {'both':>4} {'A_only':>6} {'B_only':>6} {'neither':>7} {'excl':>4} {'p_exact':>9}"
    print(header)
    print("-" * len(header))
    for run_a, run_b in args.pair:
        r = mcnemar_paired(store, run_a, run_b, condition=args.condition)
        print(
            f"{run_a:<40} {run_b:<40} {r.n_paired_items:>3} {r.both_refused:>4} {r.only_a_refused:>6} "
            f"{r.only_b_refused:>6} {r.neither_refused:>7} {r.n_excluded_pairs:>4} {r.p_value_exact:>9.2g}"
        )
    print(f"\nCondition: {args.condition}-prompt refusal. A_only = refused in run_a only.")
    print("Multiple pairs tested: apply a correction (e.g. Bonferroni: multiply p by the number of pairs).")


if __name__ == "__main__":
    main()
