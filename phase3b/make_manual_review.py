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
(Section 10): for those 40 rows body_line_by_line is left empty (a check cell to
fill), "n/a" for the other 400.

Check cells that cannot apply to a variant are prefilled "n/a" so the reviewer
only fills what is real:
  - clean: every check column (no footer, no payload).
  - neutral_en: korean_renders, receipt_copy_one_line, address_one_line.
  - neutral_ko: receipt_copy_one_line, address_one_line (Korean still checked).
  - P1_en, P2_en (arm A and arm B): korean_renders (English footer).

footer_reference carries the image's footer line(s), read from its committed
transcript (phase3/failure_cases.n_transcript_lines_for_payload, reused), joined
with " / " (empty for clean, which has no footer). image_id matches the G3(a)
results' image_id format.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import failure_cases as fc  # noqa: E402  (n_transcript_lines_for_payload, reused)

DATA_DIR = REPO_ROOT / "phase3b" / "data"
PAYLOADS_PATH = REPO_ROOT / "phase3" / "fixtures" / "payloads.json"

SAMPLE_SEED = 20261112  # Section 10, distinct from 20261108 (receipts) and 1108 (bootstrap)

ARM_A_VARIANTS = ["clean", "neutral_en", "neutral_ko", "P1_en", "P1_ko", "P2_en", "P2_ko"]
ARM_B_FONTS = ["NotoSansKR", "NanumGothic"]
ARM_B_VARIANTS = ["P1_en", "P1_ko"]
RECEIPT_IDS = [f"phase3b_eval_{i:02d}" for i in range(1, 41)]

COLUMNS = [
    "image_id", "image_path", "korean_renders", "receipt_copy_one_line",
    "address_one_line", "footer_not_cut_or_overlapping", "body_footer_spacing_ok",
    "footer_line_by_line", "body_line_by_line", "notes", "footer_reference",
]

# Reviewer check cells (notes is free text, not a check; image_path /
# footer_reference are data).
CHECK_COLUMNS = [
    "korean_renders", "receipt_copy_one_line", "address_one_line",
    "footer_not_cut_or_overlapping", "body_footer_spacing_ok",
    "footer_line_by_line", "body_line_by_line",
]


def canonical_entries() -> list[dict]:
    """The 440 entries in Section 10's canonical order, each with the arm/font/
    variant/receipt it names and its committed image/transcript paths."""
    entries: list[dict] = []
    for variant in ARM_A_VARIANTS:                      # arm A: variant x receipt
        for rid in RECEIPT_IDS:
            stem = f"{rid}_{variant}"
            entries.append({
                "image_id": f"A|{variant}|{rid}", "arm": "A", "font": None,
                "variant": variant, "receipt_id": rid,
                "image_path": f"phase3b/data/images/{stem}.png",
                "transcript_path": f"phase3b/data/transcripts/{stem}.txt",
            })
    for font in ARM_B_FONTS:                            # arm B: font x variant x receipt
        for variant in ARM_B_VARIANTS:
            for rid in RECEIPT_IDS:
                stem = f"{rid}_{variant}_{font}"
                entries.append({
                    "image_id": f"B|{font}|{variant}|{rid}", "arm": "B", "font": font,
                    "variant": variant, "receipt_id": rid,
                    "image_path": f"phase3b/data/images/{stem}.png",
                    "transcript_path": f"phase3b/data/transcripts/{stem}.txt",
                })
    assert len(entries) == 440, len(entries)
    return entries


def sampled_indices() -> list[int]:
    """The 40 body-sample indices over range(440), Section 10's fixed seed."""
    return random.Random(SAMPLE_SEED).sample(range(440), 40)


def _variant_payload(variant: str) -> tuple[str | None, str | None]:
    if variant == "clean":
        return None, None
    prefix, _, lang = variant.partition("_")
    return prefix, lang  # ("neutral"|"P1"|"P2", "en"|"ko")


def _footer_reference(transcript_rel: str, variant: str, payloads: dict) -> str:
    """The image's footer line(s), read from its committed transcript and joined
    with ' / '. Empty for clean (no footer)."""
    payload_id, lang = _variant_payload(variant)
    if payload_id is None:
        return ""
    text = (REPO_ROOT / transcript_rel).read_text(encoding="utf-8")
    k = fc.n_transcript_lines_for_payload(text, payloads[payload_id][lang])
    return " / ".join(text.splitlines()[-k:])


def _prefill_na(row: dict, variant: str) -> None:
    """Set the check cells that cannot apply to this variant to 'n/a'."""
    if variant == "clean":
        for col in CHECK_COLUMNS:
            row[col] = "n/a"
    elif variant == "neutral_en":
        for col in ("korean_renders", "receipt_copy_one_line", "address_one_line"):
            row[col] = "n/a"
    elif variant == "neutral_ko":
        for col in ("receipt_copy_one_line", "address_one_line"):
            row[col] = "n/a"
    elif variant in ("P1_en", "P2_en"):
        row["korean_renders"] = "n/a"


def build_rows() -> list[dict]:
    entries = canonical_entries()
    sampled = set(sampled_indices())
    payloads = json.loads(PAYLOADS_PATH.read_text(encoding="utf-8"))
    rows = []
    for i, e in enumerate(entries):
        row = {c: "" for c in COLUMNS}
        row["image_id"] = e["image_id"]
        row["image_path"] = e["image_path"]
        row["footer_reference"] = _footer_reference(e["transcript_path"], e["variant"], payloads)
        # body_line_by_line: empty for the 40 sampled, "n/a" for the rest.
        row["body_line_by_line"] = "" if i in sampled else "n/a"
        # variant-specific n/a prefills (clean overrides body_line_by_line too).
        _prefill_na(row, e["variant"])
        rows.append(row)
    return rows


def count_empty_per_column(rows: list[dict]) -> dict[str, int]:
    return {col: sum(1 for r in rows if r[col] == "") for col in COLUMNS}


def write_csv(path: Path) -> list[dict]:
    rows = build_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "report" / "phase3b" / "g3_manual_review.csv")
    args = parser.parse_args()
    rows = write_csv(args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print("\nEmpty cells per column:")
    for col, n in count_empty_per_column(rows).items():
        print(f"  {col:<32} {n}")


if __name__ == "__main__":
    main()
