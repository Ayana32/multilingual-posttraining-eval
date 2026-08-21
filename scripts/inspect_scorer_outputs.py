#!/usr/bin/env python3
"""Print the first N records of a run's scorer output for manual inspection.

Purpose: PolyGuardScorer.output_format_validated is False until a human has
actually read real PolyGuard raw output and confirmed the parser reads it
correctly (see mpe/scorers/polyguard.py). This script is that inspection
step -- run it right after the format-check pilot, before the full 50-item
run, per docs/phase2a-pilot.md's cloud-run checklist.

Example:
    python scripts/run_experiment.py \
        --experiment-config configs/experiments/pilot_base_en_polyguard_formatcheck.yaml \
        --evaluator hf --scorer polyguard
    python scripts/inspect_scorer_outputs.py --run-id <printed run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.storage.store import ResultStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--n", type=int, default=5, help="how many records to show")
    args = parser.parse_args()

    store = ResultStore(args.results_dir)
    records = store.read(args.run_id)
    if not records:
        print(f"No records found for run_id={args.run_id!r} in {args.results_dir}")
        return

    scored = [r for r in records if r.scorer_name is not None]
    if not scored:
        print(f"{len(records)} records found, but none have a scorer_name set "
              f"-- was this run launched with --scorer polyguard?")
        return

    print(f"{len(scored)}/{len(records)} records have scorer output. Showing first {args.n}:\n")
    for i, r in enumerate(scored[: args.n]):
        print(f"--- record {i + 1}/{min(args.n, len(scored))} ---")
        print(f"item_id:              {r.item_id}")
        print(f"expected_label:       {r.expected_label}")
        print(f"prompt:               {r.prompt[:200]!r}")
        print(f"generator completion: {r.completion[:200]!r}")
        print(f"generation_protocol:  {r.generation_protocol}")
        print(f"scorer_raw_output:    {r.scorer_raw_output!r}")
        print(f"scorer_parse_ok:      {r.scorer_parse_ok}")
        print(f"scorer_prompt_harmful:   {r.scorer_prompt_harmful}")
        print(f"scorer_response_refusal: {r.scorer_response_refusal}")
        print(f"scorer_response_harmful: {r.scorer_response_harmful}")
        print()

    parse_ok_count = sum(1 for r in scored if r.scorer_parse_ok)
    print(f"Summary: {parse_ok_count}/{len(scored)} records had scorer_parse_ok=True.")
    print(
        "Read the scorer_raw_output lines above yourself before trusting any of "
        "this -- PolyGuardScorer.output_format_validated stays False, and this "
        "integration stays 'not yet empirically validated,' until a human (not "
        "this script) has confirmed the parsed labels above actually match what "
        "the raw text says, for real generations. parse_ok=True on every record "
        "is necessary but not sufficient -- it only means the expected three "
        "lines were found, not that the parser's assumed format is the correct one."
    )


if __name__ == "__main__":
    main()
