#!/usr/bin/env python3
"""Phase 3b contact sheets for manual G3 review (docs/phase3b-replication.md
Section 10, G3(b)): one sheet for arm A (40 receipts x 7 variants = 280 cells)
and one sheet per arm B font (40 receipts x {P1_en, P1_ko} = 80 cells each, 160
total). Reuses phase3/make_contact_sheets.py's existing grid format (cell scale,
label height, label font) unchanged -- this is a review helper, not part of the
evaluation harness, and it reads the committed images via the manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import make_contact_sheets as mcs  # noqa: E402  (grid-format constants reused)

DATA_DIR = REPO_ROOT / "phase3b" / "data"
ARM_A_VARIANTS = ["clean", "neutral_en", "neutral_ko", "P1_en", "P1_ko", "P2_en", "P2_ko"]
ARM_B_VARIANTS = ["P1_en", "P1_ko"]
ARM_B_FONTS = ["NotoSansKR", "NanumGothic"]


def load_manifest(data_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def _build_grid(rows: list[dict], col_labels: list[str], data_dir: Path, out_path: Path, title: str) -> dict:
    """Grid of receipt_id (rows) x col_labels (columns), same visual format as
    phase3/make_contact_sheets.py. rows must each carry receipt_id, variant and
    image_path (relative to data_dir)."""
    assert rows, f"no rows for {title}"
    receipt_ids = sorted({r["receipt_id"] for r in rows})
    by_key = {(r["receipt_id"], r["variant"]): r for r in rows}

    label_font = ImageFont.truetype(str(mcs.FONTS_DIR / "NotoSans-Variable.ttf"), mcs.LABEL_FONT_SIZE)
    label_font.set_variation_by_name("Regular")

    sample = Image.open(data_dir / by_key[(receipt_ids[0], col_labels[0])]["image_path"])
    cell_w = int(sample.width * mcs.CELL_SCALE)
    cell_h = int(sample.height * mcs.CELL_SCALE)
    block_w = cell_w + mcs.CELL_PADDING
    block_h = cell_h + mcs.LABEL_HEIGHT + mcs.CELL_PADDING

    n_rows, n_cols = len(receipt_ids), len(col_labels)
    sheet_w = n_cols * block_w + mcs.CELL_PADDING
    sheet_h = n_rows * block_h + mcs.CELL_PADDING
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    cells = 0
    for row_i, rid in enumerate(receipt_ids):
        for col_i, variant in enumerate(col_labels):
            x = mcs.CELL_PADDING + col_i * block_w
            y = mcs.CELL_PADDING + row_i * block_h
            row = by_key.get((rid, variant))
            if row is None:
                draw.text((x, y), f"{rid}\n{variant}\nMISSING", font=label_font, fill="red")
                continue
            img = Image.open(data_dir / row["image_path"]).resize((cell_w, cell_h), Image.LANCZOS)
            sheet.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline="lightgray")
            draw.text((x, y + cell_h + 2), f"{rid} {variant}", font=label_font, fill="black")
            cells += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"{title}: {n_rows} receipts x {n_cols} cols -> {out_path} ({sheet_w}x{sheet_h}), {cells} cells")
    return {"n_rows": n_rows, "n_cols": n_cols, "cells": cells, "size": (sheet_w, sheet_h)}


def build_arm_a_sheet(data_dir: Path, out_path: Path) -> dict:
    manifest = load_manifest(data_dir)
    rows = [r for r in manifest if r.get("arm") == "A"]
    return _build_grid(rows, ARM_A_VARIANTS, data_dir, out_path, "arm A")


def build_arm_b_sheet(data_dir: Path, font_label: str, out_path: Path) -> dict:
    manifest = load_manifest(data_dir)
    rows = [r for r in manifest if r.get("arm") == "B" and r.get("font") == font_label]
    return _build_grid(rows, ARM_B_VARIANTS, data_dir, out_path, f"arm B {font_label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "contact_sheets")
    args = parser.parse_args()
    build_arm_a_sheet(args.data_dir, args.out_dir / "contact_sheet_arm_a.png")
    for font_label in ARM_B_FONTS:
        build_arm_b_sheet(args.data_dir, font_label, args.out_dir / f"contact_sheet_arm_b_{font_label}.png")


if __name__ == "__main__":
    main()
