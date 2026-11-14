#!/usr/bin/env python3
"""Build report/phase3b/g3_manual_review.csv: the 440-row manual G3 (b)/(c)
review sheet, in docs/phase3b-replication.md Section 10's canonical order.

Canonical order (Section 10, the same ordering the body-sample seed draws over):
  - arm A: variant order clean, neutral_en, neutral_ko, P1_en, P1_ko, P2_en,
    P2_ko, each x phase3b_eval_01..40  (280 rows)
  - arm B: font order NotoSansKR, NanumGothic, each x variant P1_en, P1_ko,
    each x phase3b_eval_01..40  (160 rows)

The 40 body-line-by-line sampled images are
random.Random(20261112).sample(range(440), 40) over that canonical ordering
(Section 10). For those 40 rows the body_line_by_line cell is left empty (a
check cell to fill in review); for the other 400 it is "n/a" (body not sampled).
Every other check cell is left empty for the reviewer. image_id matches the G3(a)
results' image_id format.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_SEED = 20261112  # Section 10, distinct from 20261108 (receipts) and 1108 (bootstrap)

ARM_A_VARIANTS = ["clean", "neutral_en", "neutral_ko", "P1_en", "P1_ko", "P2_en", "P2_ko"]
ARM_B_FONTS = ["NotoSansKR", "NanumGothic"]
ARM_B_VARIANTS = ["P1_en", "P1_ko"]
RECEIPT_IDS = [f"phase3b_eval_{i:02d}" for i in range(1, 41)]

COLUMNS = [
    "image_id", "image_path", "korean_renders", "receipt_copy_one_line",
    "address_one_line", "footer_not_cut_or_overlapping", "body_footer_spacing_ok",
    "footer_line_by_line", "body_line_by_line", "notes",
]


def canonical_entries() -> list[dict]:
    """The 440 (image_id, image_path) entries in Section 10's canonical order."""
    entries: list[dict] = []
    # arm A: variant x receipt
    for variant in ARM_A_VARIANTS:
        for rid in RECEIPT_IDS:
            entries.append({
                "image_id": f"A|{variant}|{rid}",
                "image_path": f"phase3b/data/images/{rid}_{variant}.png",
            })
    # arm B: font x variant x receipt
    for font in ARM_B_FONTS:
        for variant in ARM_B_VARIANTS:
            for rid in RECEIPT_IDS:
                entries.append({
                    "image_id": f"B|{font}|{variant}|{rid}",
                    "image_path": f"phase3b/data/images/{rid}_{variant}_{font}.png",
                })
    assert len(entries) == 440, len(entries)
    return entries


def sampled_indices() -> list[int]:
    """The 40 body-sample indices over range(440), Section 10's fixed seed."""
    return random.Random(SAMPLE_SEED).sample(range(440), 40)


def build_rows() -> list[dict]:
    entries = canonical_entries()
    sampled = set(sampled_indices())
    rows = []
    for i, e in enumerate(entries):
        row = {c: "" for c in COLUMNS}
        row["image_id"] = e["image_id"]
        row["image_path"] = e["image_path"]
        # body_line_by_line: empty for the 40 sampled, "n/a" for the rest.
        row["body_line_by_line"] = "" if i in sampled else "n/a"
        rows.append(row)
    return rows


def write_csv(path: Path) -> int:
    rows = build_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "report" / "phase3b" / "g3_manual_review.csv")
    args = parser.parse_args()
    n = write_csv(args.out)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
