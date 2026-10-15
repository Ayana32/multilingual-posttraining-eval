#!/usr/bin/env python3
"""Compare 200-token vs 1024-token pilot runs for the length ablation.
See docs/phase2a-length-ablation.md for the preregistered plan this
implements. Deterministic: fixed bootstrap seed, no LLM, no network, no
model inference -- reads only already-stored ResultRecords.

Example (pairs file, the normal batch mode -- needs one EN and one KO
pair per stage to compute cross-language gates/outcomes):
    python scripts/compare_length_ablation.py --pairs-file pairs.json

Example (single pair, e.g. for a quick ad-hoc check):
    python scripts/compare_length_ablation.py \
        --stage sft --language en \
        --run-200 pilot_sft_en_polyguard_797e5b18 \
        --run-1024 <new-1024-run-id>

pairs.json format:
    {"pairs": [
        {"stage": "sft", "language": "en", "run_200": "...", "run_1024": "..."},
        {"stage": "sft", "language": "ko", "run_200": "...", "run_1024": "..."},
        ...
    ]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.analysis.length_ablation import (  # noqa: E402
    LengthAblationPair,
    build_report,
    extract_review_candidates,
    format_report_text,
    write_review_candidates_csv,
    write_review_key_csv,
)
from mpe.storage.store import ResultStore  # noqa: E402


def _load_pairs(args: argparse.Namespace) -> list[LengthAblationPair]:
    if args.pairs_file:
        data = json.loads(Path(args.pairs_file).read_text())
        return [LengthAblationPair(**p) for p in data["pairs"]]
    if args.stage and args.language and args.run_200 and args.run_1024:
        return [
            LengthAblationPair(
                stage=args.stage, language=args.language, run_200=args.run_200, run_1024=args.run_1024
            )
        ]
    raise SystemExit("Provide either --pairs-file, or all of --stage/--language/--run-200/--run-1024.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--stage", choices=["base", "sft", "dpo", "rlvr"], default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--run-200", default=None)
    parser.add_argument("--run-1024", default=None)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-out", type=Path, default=None, help="Optional path to also write the human-readable report."
    )
    parser.add_argument(
        "--gate-verdict-out", type=Path, default=None, help="Optional path to write the machine-readable gate verdict JSON."
    )
    parser.add_argument(
        "--review-csv-out", type=Path, default=None,
        help="Optional path to write the annotator-facing manual-review CSV (no reasons column).",
    )
    parser.add_argument(
        "--review-key-csv-out", type=Path, default=None,
        help="Optional path to write the analyst-only key CSV (stage, language, parallel_item_id -> "
        "reasons), kept separate so annotators stay blind to scorer-derived flags.",
    )
    args = parser.parse_args()

    pairs = _load_pairs(args)
    store = ResultStore(args.results_dir)

    report = build_report(store, pairs, n_resamples=args.n_resamples, seed=args.seed)
    text = format_report_text(report)
    print(text)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text)
        print(f"\nReport written to {args.report_out}")

    if args.gate_verdict_out:
        args.gate_verdict_out.parent.mkdir(parents=True, exist_ok=True)
        args.gate_verdict_out.write_text(report.gate_verdict.model_dump_json(indent=2))
        print(f"Gate verdict written to {args.gate_verdict_out}")

    candidates = extract_review_candidates(store, pairs)
    if args.review_csv_out:
        path = write_review_candidates_csv(candidates, args.review_csv_out)
        print(f"{len(candidates)} review candidates written to {path} (annotator-facing, no reasons column)")
        if args.review_key_csv_out:
            key_path = write_review_key_csv(candidates, args.review_key_csv_out)
            print(f"Reasons key (analyst-only) written to {key_path}")
    else:
        print(f"\n{len(candidates)} manual-review candidates found (pass --review-csv-out to write them to a file)")


if __name__ == "__main__":
    main()
