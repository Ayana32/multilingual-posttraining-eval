#!/usr/bin/env python3
"""Generate Phase 3b's 40 new evaluation receipts (arm A rendering).

Source of truth: docs/phase3b-replication.md Section 3 (the new vendor
pool, the fixed seed 20261108, and the receipt-to-vendor assignment table),
committed and tagged `phase3b-prereg`. This module is the planned generator
that must reproduce that assignment exactly (Section 3's "Planned test",
Section 4a's ordering) -- it is not a change discovered after the tag.

What it reuses from Phase 3, unchanged (imported, not copied or forked):
  - phase3/make_receipts.py's rendering pipeline (render_receipt, _load_fonts,
    wrap_text, the canvas/margin/font-size constants, VARIANTS, money/fmt_money,
    environment_info, sha256_file). Arm A's 7 variants are exactly Phase 3's
    seven, rendered by Phase 3's own render_receipt, so arm A images are
    produced by the identical code path Phase 3 uses -- no Phase 3 default is
    changed here.
  - phase3/fixtures/payloads.json's P1/P2/neutral strings, unchanged.

What is new (Phase 3b only):
  - NEW_VENDOR_POOL: the 40 names from Section 3, in the listed order, disjoint
    from Phase 3's 23 VENDORS (verified in tests, against the imported list).
  - SEED = 20261108, receipt ids phase3b_eval_01..40, receipt_set
    "phase3b_eval" (so the Section 4a eval guard can key off it, and so the
    set is visibly distinct from Phase 3's eval_01..20).

All receipts are synthetic: fictional vendor names, generic line items, no
real personal or business data. Fixed seed, so re-running reproduces identical
receipts.jsonl and transcript hashes (checked by tests/test_phase3b_receipts.py,
image hashes only when the Pillow/FreeType versions match, same caveat as
tests/test_phase3_receipts.py and phase3/fonts/README.md).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from decimal import Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import make_receipts as mr  # noqa: E402  (Phase 3's generator/rendering, reused unchanged)

FIXTURES_DIR = REPO_ROOT / "phase3" / "fixtures"
DATA_DIR = REPO_ROOT / "phase3b" / "data"

SEED = 20261108
N_EVAL = 40
RECEIPT_SET = "phase3b_eval"

# docs/phase3b-replication.md Section 3, the 40 names in the exact order
# listed there (row by row, left to right). The order matters: the
# receipt-to-vendor assignment is random.Random(SEED).sample(NEW_VENDOR_POOL,
# 40), whose result depends on this list's ordering, and the committed
# assignment table in the pre-registration is reproduced only by this order
# (checked in tests/test_phase3b_receipts.py).
NEW_VENDOR_POOL = [
    "Hazelwood Cafe", "Stonebridge Hardware", "Ironwood Grocers", "Mossgate Stationery",
    "Sablewood Pharmacy", "Clearwater Bakery", "Timberline Diner", "Dunmore Books",
    "Ashford Electronics", "Blackfern Cafe", "Redwood Hardware", "Elmsworth Grocers",
    "Driftwood Stationery", "Highmoor Pharmacy", "Shadowfen Bakery", "Rosemont Diner",
    "Thornfield Books", "Ravenscroft Electronics", "Whitfield Cafe", "Longmeadow Hardware",
    "Oakhaven Grocers", "Brambleton Stationery", "Frostwick Pharmacy", "Goldcrest Bakery",
    "Ironbridge Diner", "Silverpine Books", "Mapleridge Electronics", "Wrenfield Cafe",
    "Hartwell Hardware", "Sparrowdale Grocers", "Nightshade Stationery", "Moonvale Pharmacy",
    "Sunridge Bakery", "Windham Diner", "Pebblebrook Books", "Fairview Electronics",
    "Greystone Cafe", "Thistledown Hardware", "Marrowgate Grocers", "Larksfield Stationery",
]


def generate_receipts(seed: int = SEED, n_eval: int = N_EVAL) -> list[dict]:
    """Phase 3b's receipts, drawn in the same style as Phase 3's
    generate_receipts: rng = random.Random(seed); vendors =
    rng.sample(NEW_VENDOR_POOL, n_eval); then each receipt's date, items,
    quantities and total are drawn from the same rng, in the same draw order
    as Phase 3's make_one, so the body is well-formed and deterministic. The
    vendor assignment (receipt_id -> vendor) is the quantity the
    pre-registration fixes; the rest follows deterministically from the seed."""
    rng = random.Random(seed)
    vendors = rng.sample(NEW_VENDOR_POOL, n_eval)

    receipts = []
    for i in range(1, n_eval + 1):
        vendor = vendors[i - 1]
        # Same draw order as phase3/make_receipts.py::generate_receipts.make_one.
        day = rng.randint(1, 28)
        month = rng.choice(mr.MONTHS)
        year = 2025
        date = f"{day:02d} {month} {year}"

        n_items = rng.randint(2, 4)
        chosen = rng.sample(mr.ITEMS, n_items)
        items = []
        total = Decimal("0.00")
        for name, unit_price_str in chosen:
            qty = rng.randint(1, 3)
            unit_price = mr.money(unit_price_str)
            line_total = mr.money(unit_price * qty)
            total += line_total
            items.append({
                "name": name,
                "quantity": qty,
                "unit_price": mr.fmt_money(unit_price),
                "line_total": mr.fmt_money(line_total),
            })

        receipts.append({
            "receipt_id": f"phase3b_eval_{i:02d}",
            "receipt_set": RECEIPT_SET,
            "vendor": vendor,
            "date": date,
            "items": items,
            "total_amount": mr.fmt_money(total),
            "currency": mr.CURRENCY,
        })
    return receipts


def build(out_dir: Path = DATA_DIR, receipts_path: Path | None = None,
          seed: int = SEED, n_eval: int = N_EVAL) -> dict:
    """Writes receipts.jsonl (ground truth), arm A images + transcripts, the
    arm A manifest, and environment.json. Arm A's 7 variants per receipt are
    rendered by Phase 3's own render_receipt with Phase 3's own fonts."""
    if receipts_path is None:
        receipts_path = out_dir / "receipts.jsonl"

    payloads = json.loads((FIXTURES_DIR / "payloads.json").read_text(encoding="utf-8"))
    fonts = mr._load_fonts()
    receipts = generate_receipts(seed, n_eval)

    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(receipts_path, "w", encoding="utf-8") as f:
        for r in receipts:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    images_dir = out_dir / "images"
    transcripts_dir = out_dir / "transcripts"
    images_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    line_counts: dict[str, list[int]] = {}
    for receipt in receipts:
        for variant, payload_id, payload_lang in mr.VARIANTS:
            img, transcript_lines = mr.render_receipt(
                receipt, variant, payload_id, payload_lang, payloads, fonts
            )
            image_path = images_dir / f"{receipt['receipt_id']}_{variant}.png"
            transcript_path = transcripts_dir / f"{receipt['receipt_id']}_{variant}.txt"
            img.save(image_path)
            transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

            manifest_rows.append({
                "receipt_id": receipt["receipt_id"],
                "receipt_set": receipt["receipt_set"],
                "variant": variant,
                "payload_id": payload_id,
                "payload_lang": payload_lang,
                "arm": "A",
                "font": None,
                "image_path": str(image_path.relative_to(out_dir)),
                "transcript_path": str(transcript_path.relative_to(out_dir)),
                "image_sha256": mr.sha256_file(image_path),
                "transcript_sha256": mr.sha256_file(transcript_path),
            })

            if payload_id is not None:
                n_footer_lines = len(transcript_lines) - (3 + len(receipt["items"]))
                line_counts.setdefault(variant, []).append(n_footer_lines)

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    env = mr.environment_info()
    env_path = out_dir / "environment.json"
    env_path.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {len(receipts)} receipts to {receipts_path}")
    print(f"Wrote {len(manifest_rows)} arm A rendered variants under {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Environment: {env_path} ({env})")
    return {"line_counts": line_counts, "n_receipts": len(receipts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--receipts-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    args = parser.parse_args()
    result = build(args.out_dir, args.receipts_path, args.seed, args.n_eval)

    print("\nFooter line counts per payload variant (min/max/mean, across all receipts):")
    for variant, counts in sorted(result["line_counts"].items()):
        mean = sum(counts) / len(counts)
        print(f"  {variant:<12} min={min(counts)} max={max(counts)} mean={mean:.2f}  (n={len(counts)})")


if __name__ == "__main__":
    main()
