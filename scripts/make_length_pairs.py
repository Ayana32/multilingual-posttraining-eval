#!/usr/bin/env python3
"""Build the length-ablation pairs.json automatically instead of by hand.
See docs/phase2a-length-ablation.md section 1 for the six (stage,
language) conditions and their 200-token run IDs.

For each of the six (stage, language) combinations: the 200-token run is
looked up by its known, hardcoded run ID (KNOWN_VALID_RUN_IDS_200 below,
copied from docs/phase2a-length-ablation.md section 1 -- keep the two in
sync if either changes); the 1024-token run is discovered in --results-dir
by experiment_name == "pilot_{stage}_{lang}_polyguard_len1024". Every run
used (200 and 1024 side) is validated against KNOWN_INVALID_RUN_IDS and
against living directly under --results-dir (not under results/_invalid/)
before being written out -- this fails loudly rather than silently
building a pairs file around a known-bad or relocated run.

No model inference, no network. Reads only already-stored ResultRecords.

Example:
    python scripts/make_length_pairs.py --out pairs.json
    python scripts/compare_length_ablation.py --pairs-file pairs.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.analysis.length_ablation import KNOWN_INVALID_RUN_IDS  # noqa: E402
from mpe.storage.store import ResultStore  # noqa: E402

STAGES = ["sft", "dpo", "rlvr"]
LANGUAGES = ["en", "ko"]

# The six valid 200-token pilot run IDs -- see
# docs/phase2a-length-ablation.md section 1. Hardcoded deliberately: these
# are specific, already-run, already-audited run directories, not
# something to rediscover by experiment_name (which could also match a
# stale re-run or a format-check run on the same experiment name).
KNOWN_VALID_RUN_IDS_200: dict[tuple[str, str], str] = {
    ("sft", "en"): "pilot_sft_en_polyguard_797e5b18",
    ("sft", "ko"): "pilot_sft_ko_polyguard_b200d0c5",
    ("dpo", "en"): "pilot_dpo_en_polyguard_e73913f7",
    ("dpo", "ko"): "pilot_dpo_ko_polyguard_071840f2",
    ("rlvr", "en"): "pilot_rlvr_en_polyguard_152ef3e1",
    ("rlvr", "ko"): "pilot_rlvr_ko_polyguard_74628a89",
}


class PairDiscoveryError(RuntimeError):
    pass


def _validate_run_usable(results_dir: Path, run_id: str) -> None:
    """Fails loudly if `run_id` is known-bad or doesn't live where
    expected. Applied to every run (200 side and 1024 side) before it is
    written into the pairs file."""
    if run_id in KNOWN_INVALID_RUN_IDS:
        raise PairDiscoveryError(
            f"Run '{run_id}' is in KNOWN_INVALID_RUN_IDS (see mpe.analysis.length_ablation) and must "
            "never be used."
        )
    if (results_dir / "_invalid" / run_id).exists():
        raise PairDiscoveryError(
            f"Run '{run_id}' exists under {results_dir / '_invalid'} -- known-bad, must never be used."
        )
    if not (results_dir / run_id / "records.jsonl").exists():
        raise PairDiscoveryError(
            f"Run '{run_id}' not found directly under {results_dir} "
            f"({results_dir / run_id / 'records.jsonl'} does not exist)."
        )


def _find_1024_run(store: ResultStore, stage: str, language: str) -> str:
    experiment_name = f"pilot_{stage}_{language}_polyguard_len1024"
    matches = []
    for run_id in store.list_runs():
        records = store.read(run_id)
        if records and records[0].experiment_name == experiment_name:
            matches.append(run_id)

    if len(matches) == 0:
        raise PairDiscoveryError(
            f"No run found for experiment '{experiment_name}' under {store.results_dir}. Run the "
            f"corresponding pilot_{stage}_{language}_polyguard_len1024.yaml config first."
        )
    if len(matches) > 1:
        raise PairDiscoveryError(
            f"Expected exactly one run for experiment '{experiment_name}', found {len(matches)}: "
            f"{sorted(matches)}. Remove or relocate the extras (e.g. to results/_invalid/) before "
            "building pairs.json -- an ambiguous match is not a safe default to silently pick from."
        )
    return matches[0]


def build_pairs(results_dir: str | Path) -> list[dict[str, str]]:
    results_dir = Path(results_dir)
    store = ResultStore(results_dir)
    pairs = []
    for stage in STAGES:
        for language in LANGUAGES:
            run_200 = KNOWN_VALID_RUN_IDS_200[(stage, language)]
            _validate_run_usable(results_dir, run_200)

            run_1024 = _find_1024_run(store, stage, language)
            _validate_run_usable(results_dir, run_1024)

            pairs.append({"stage": stage, "language": language, "run_200": run_200, "run_1024": run_1024})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Path to write the pairs.json file to.")
    args = parser.parse_args()

    pairs = build_pairs(args.results_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"pairs": pairs}, indent=2) + "\n")
    print(f"Wrote {len(pairs)} pairs to {args.out}")
    for p in pairs:
        print(f"  {p['stage']} {p['language']}: {p['run_200']} <-> {p['run_1024']}")


if __name__ == "__main__":
    main()
