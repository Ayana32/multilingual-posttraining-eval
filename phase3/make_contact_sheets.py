#!/usr/bin/env python3
"""Build one contact-sheet image per receipt set (tune, eval), for manual
G3 review (docs/phase3-visual-injection.md section 5). Each sheet is a
grid of every rendered variant for every receipt in that set, read from
the manifest produced by phase3/make_receipts.py. This is a review
helper only, not part of the evaluation harness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "phase3" / "data"
FONTS_DIR = REPO_ROOT / "phase3" / "fonts"

VARIANT_ORDER = ["clean", "neutral_en", "neutral_ko", "P1_en", "P1_ko", "P2_en", "P2_ko"]

CELL_SCALE = 0.7
LABEL_HEIGHT = 20
LABEL_FONT_SIZE = 13
CELL_PADDING = 10


def _load_manifest(data_dir: Path) -> list[dict]:
    return [json.loads(line) for line in open(data_dir / "manifest.jsonl", encoding="utf-8") if line.strip()]


def build_contact_sheet(receipt_set: str, data_dir: Path, out_path: Path) -> None:
    manifest = _load_manifest(data_dir)
    rows_data = [r for r in manifest if r["receipt_set"] == receipt_set]
    assert rows_data, f"no manifest rows found for receipt_set={receipt_set!r}"

    receipt_ids = sorted({r["receipt_id"] for r in rows_data})
    by_key = {(r["receipt_id"], r["variant"]): r for r in rows_data}

    label_font = ImageFont.truetype(str(FONTS_DIR / "NotoSans-Variable.ttf"), LABEL_FONT_SIZE)
    label_font.set_variation_by_name("Regular")

    # Determine cell size from the first available image.
    sample_img = Image.open(data_dir / by_key[(receipt_ids[0], VARIANT_ORDER[0])]["image_path"])
    cell_w = int(sample_img.width * CELL_SCALE)
    cell_h = int(sample_img.height * CELL_SCALE)
    block_w = cell_w + CELL_PADDING
    block_h = cell_h + LABEL_HEIGHT + CELL_PADDING

    n_rows = len(receipt_ids)
    n_cols = len(VARIANT_ORDER)
    sheet_w = n_cols * block_w + CELL_PADDING
    sheet_h = n_rows * block_h + CELL_PADDING

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row_i, receipt_id in enumerate(receipt_ids):
        for col_i, variant in enumerate(VARIANT_ORDER):
            row = by_key.get((receipt_id, variant))
            x = CELL_PADDING + col_i * block_w
            y = CELL_PADDING + row_i * block_h
            if row is None:
                draw.text((x, y), f"{receipt_id}\n{variant}\nMISSING", font=label_font, fill="red")
                continue
            img = Image.open(data_dir / row["image_path"]).resize((cell_w, cell_h), Image.LANCZOS)
            sheet.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline="lightgray")
            draw.text((x, y + cell_h + 2), f"{receipt_id} {variant}", font=label_font, fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"{receipt_set}: {n_rows} receipts x {n_cols} variants -> {out_path} ({sheet_w}x{sheet_h})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "contact_sheets")
    args = parser.parse_args()

    for receipt_set in ["tune", "eval"]:
        out_path = args.out_dir / f"contact_sheet_{receipt_set}.png"
        build_contact_sheet(receipt_set, args.data_dir, out_path)


if __name__ == "__main__":
    main()
