#!/usr/bin/env python3
"""Build redacted, publishable copies of the Phase 2B records.jsonl files.

Reads real results (prompt, completion, scorer_raw_output included) from a
raw results directory given on the command line, and writes one
records_redacted.jsonl per run under data/phase2b/<run_id>/, keeping every
field the committed analysis scripts need and dropping prompt, completion,
and scorer_raw_output. completion_sha256, completion_char_len, and three
repetition metrics (dup_fraction, max_5gram_count, is_collapse) are computed
from the completion text before it is dropped, so downstream analysis never
needs the raw text again.

This is the one script in the repo that touches raw completions; everything
else reads only the redacted output this script produces. Read-only on the
raw results directory; writes only under data/phase2b/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

# (stage, language) -> run_id, matching docs/phase2b-results.md section 2.
PHASE2B_RUNS = {
    ("dpo", "en"): "phase2b_dpo_en_polyguard_17c33a34",
    ("dpo", "ko"): "phase2b_dpo_ko_polyguard_7b95ba0c",
    ("rlvr", "en"): "phase2b_rlvr_en_polyguard_151e7314",
    ("rlvr", "ko"): "phase2b_rlvr_ko_polyguard_324c0cc3",
}

KEPT_FIELDS = [
    "run_id", "stage", "language", "item_id", "parallel_item_id",
    "expected_label", "finish_reason", "language_match", "refusal_label",
    "behavior_matches_expected", "scorer_name", "scorer_prompt_harmful",
    "scorer_response_refusal", "scorer_response_harmful", "scorer_parse_ok",
    "generation_config", "hf_repo_id", "revision", "created_at",
]


def sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def word_ngrams(text: str, n: int) -> list[tuple]:
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def repetition_score(text: str) -> dict:
    """dup_fraction (repeated-sentence fraction) and max_5gram_count (worst
    repeated 5-gram of words) together flag degenerate, looping generation.
    is_collapse is the same threshold used throughout this project's
    analysis: dup_fraction > 0.3 or max_5gram_count >= 3."""
    sents = sentences(text)
    n_sents = len(sents)
    dup_sents = n_sents - len(set(sents)) if n_sents else 0
    dup_fraction = dup_sents / n_sents if n_sents else 0.0

    grams5 = word_ngrams(text, 5)
    max_5gram_count = max(Counter(grams5).values()) if grams5 else 0

    is_collapse = dup_fraction > 0.3 or max_5gram_count >= 3
    return {
        "dup_fraction": dup_fraction,
        "max_5gram_count": max_5gram_count,
        "is_collapse": is_collapse,
    }


def redact_record(record: dict) -> dict:
    completion = record["completion"]
    out = {field: record[field] for field in KEPT_FIELDS}
    out["completion_sha256"] = hashlib.sha256(completion.encode("utf-8")).hexdigest()
    out["completion_char_len"] = len(completion)
    out.update(repetition_score(completion))
    return out


def build_redacted_records(raw_results_dir: Path, out_dir: Path) -> dict[str, int]:
    counts = {}
    for (stage, lang), run_id in PHASE2B_RUNS.items():
        src = raw_results_dir / run_id / "records.jsonl"
        records = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]

        dst_dir = out_dir / run_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "records_redacted.jsonl"
        with open(dst, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(redact_record(r), ensure_ascii=False, sort_keys=True) + "\n")
        counts[run_id] = len(records)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_results_dir", type=Path, help="Path to the raw results directory (e.g. ~/results)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase2b"))
    args = parser.parse_args()

    raw_results_dir = args.raw_results_dir.expanduser()
    counts = build_redacted_records(raw_results_dir, args.out_dir)
    for run_id, n in counts.items():
        print(f"{run_id}: {n} records redacted -> {args.out_dir / run_id / 'records_redacted.jsonl'}")


if __name__ == "__main__":
    main()
