#!/usr/bin/env python3
"""One-off: build report/human_labels_en.csv or report/human_labels_ko.csv
from a completed annotation sheet, joined against its key CSV.

The output has exactly: annotation_id, item_id, parallel_item_id, stage,
human_refusal, human_harmful, notes. No prompt or completion column, so
this is the file committed to the public repo; the completed sheet itself
(which does have prompt/completion) is never committed.

Handles a completed sheet that has a stray title row as its first line
(e.g. a spreadsheet export that added "en_scorer_validation" above the
real header) by detecting and dropping it automatically.

If the key CSV doesn't carry parallel_item_id directly (the original KO
key CSV only has item_id), it is resolved from the redacted Phase 2B
records under --data-dir instead of raw results, so this script never
needs raw completions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_COMPLETED_COLS = {"annotation_id", "prompt", "completion", "finish_reason",
                            "human_refusal", "human_harmful", "notes"}


def read_completed_sheet(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    first_line, _, rest = text.partition("\n")
    if "annotation_id" not in first_line:
        # Stray title row (e.g. a spreadsheet export's sheet-name line); drop it.
        text = rest
    rows = list(csv.DictReader(text.splitlines()))
    assert rows, f"no rows parsed from {path}"
    assert EXPECTED_COMPLETED_COLS.issubset(rows[0].keys()), rows[0].keys()
    return rows


def yn(v: str) -> bool:
    return v.strip().lower() == "yes"


def build_item_id_to_pid(data_dir: Path) -> dict[str, str]:
    mapping = {}
    for records_path in sorted(data_dir.glob("*/records_redacted.jsonl")):
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    mapping[r["item_id"]] = r["parallel_item_id"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--completed-sheet", required=True, type=Path)
    parser.add_argument("--key-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/phase2b"),
                         help="Used only to resolve parallel_item_id when the key CSV lacks it.")
    args = parser.parse_args()

    completed_rows = read_completed_sheet(args.completed_sheet)
    key_rows = list(csv.DictReader(open(args.key_csv, encoding="utf-8-sig")))
    key_by_id = {r["annotation_id"]: r for r in key_rows}
    assert {r["annotation_id"] for r in completed_rows} == set(key_by_id), "annotation_id mismatch"

    item_id_to_pid = None
    if key_rows and "parallel_item_id" not in key_rows[0]:
        item_id_to_pid = build_item_id_to_pid(args.data_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["annotation_id", "item_id", "parallel_item_id", "stage",
                                                 "human_refusal", "human_harmful", "notes"])
        writer.writeheader()
        for c in completed_rows:
            k = key_by_id[c["annotation_id"]]
            pid = k.get("parallel_item_id") or (item_id_to_pid or {})[k["item_id"]]
            writer.writerow({
                "annotation_id": c["annotation_id"],
                "item_id": k["item_id"],
                "parallel_item_id": pid,
                "stage": k["stage"],
                "human_refusal": str(yn(c["human_refusal"])),
                "human_harmful": str(yn(c["human_harmful"])),
                "notes": c["notes"],
            })
    print(f"Wrote {len(completed_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
