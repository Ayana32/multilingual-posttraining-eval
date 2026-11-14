#!/usr/bin/env python3
"""Phase 3b arm B rendering: font sensitivity (docs/phase3b-replication.md
Section 7).

Arm B re-renders P1_en and P1_ko (image channel only, the same 40 receipts as
arm A) under each of two single-file fonts that cover both Latin and Hangul,
used for BOTH scripts instead of Phase 3's script-switched pair:

  - "NotoSansKR": the same NotoSansKR-Variable.ttf Regular instance Phase 3
    already uses for its Korean footer. Reused as the exact same loaded font
    object Phase 3's arm A uses for ko_footer, so P1_ko under this candidate is
    byte-identical to arm A's P1_ko (Section 7's determinism check).
  - "NanumGothic": NanumGothic-Regular.ttf, a STATIC font (no fvar). Per
    Section 7, set_variation_by_name is NOT called on it; it is loaded with
    layout_engine=BASIC at size 16 and nothing more.

Only the footer font changes. Body and title stay in Phase 3's Noto Sans at
their Phase 3 sizes, and the whole render otherwise goes through Phase 3's own
render_receipt unchanged -- which is what makes the P1_ko/NotoSansKR identity
exact rather than approximate. The footer font for both en and ko is set to the
single candidate font, so render_receipt (which picks en_footer for lang=="en"
and ko_footer otherwise) uses the candidate regardless of language.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import make_receipts as mr  # noqa: E402  (Phase 3 rendering, reused unchanged)

# phase3b.make_receipts, loaded by path to avoid the make_receipts name clash.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "phase3b_make_receipts", REPO_ROOT / "phase3b" / "make_receipts.py"
)
p3b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3b)

FONTS_DIR = REPO_ROOT / "phase3" / "fonts"
FIXTURES_DIR = REPO_ROOT / "phase3" / "fixtures"
DATA_DIR = REPO_ROOT / "phase3b" / "data"

ARM_B_FONTS = ["NotoSansKR", "NanumGothic"]
# (variant, payload_id, payload_lang), image channel only.
ARM_B_VARIANTS = [("P1_en", "P1", "en"), ("P1_ko", "P1", "ko")]


def load_arm_b_fonts() -> dict[str, dict]:
    """One `fonts` dict per candidate, each in the shape Phase 3's
    render_receipt expects (en_body, en_title, en_footer, ko_footer). Body and
    title are Phase 3's exact Noto Sans objects; en_footer and ko_footer are
    BOTH the candidate footer font, so the single font is used for both scripts.

    For NotoSansKR the footer is the *same* object Phase 3 uses for ko_footer
    (reused from mr._load_fonts()), which is what makes P1_ko byte-identical to
    arm A. For NanumGothic the footer is loaded fresh WITHOUT
    set_variation_by_name (it is a static font; Section 7)."""
    base = mr._load_fonts()

    nanum_footer = ImageFont.truetype(
        str(FONTS_DIR / "NanumGothic-Regular.ttf"),
        mr.FOOTER_FONT_SIZE,
        layout_engine=ImageFont.Layout.BASIC,
    )
    # Deliberately no set_variation_by_name: NanumGothic-Regular.ttf has no fvar.

    notokr_footer = base["ko_footer"]  # the exact object arm A uses for ko_footer

    common = {"en_body": base["en_body"], "en_title": base["en_title"]}
    return {
        "NotoSansKR": {**common, "en_footer": notokr_footer, "ko_footer": notokr_footer},
        "NanumGothic": {**common, "en_footer": nanum_footer, "ko_footer": nanum_footer},
    }


def arm_b_manifest_rows(out_dir: Path, seed: int = p3b.SEED, n_eval: int = p3b.N_EVAL,
                        write: bool = True) -> list[dict]:
    """Render arm B's images/transcripts (P1_en, P1_ko x 40 receipts x 2 fonts =
    160 images) under out_dir and return their manifest rows (arm "B", with the
    font label). Filenames carry the font label so arm A and the two arm B
    candidates never collide: {receipt_id}_{variant}_{font}.png."""
    payloads = json.loads((FIXTURES_DIR / "payloads.json").read_text(encoding="utf-8"))
    fonts_by_label = load_arm_b_fonts()
    receipts = p3b.generate_receipts(seed, n_eval)

    images_dir = out_dir / "images"
    transcripts_dir = out_dir / "transcripts"
    if write:
        images_dir.mkdir(parents=True, exist_ok=True)
        transcripts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for font_label in ARM_B_FONTS:
        fonts = fonts_by_label[font_label]
        for receipt in receipts:
            for variant, payload_id, payload_lang in ARM_B_VARIANTS:
                img, transcript_lines = mr.render_receipt(
                    receipt, variant, payload_id, payload_lang, payloads, fonts
                )
                stem = f"{receipt['receipt_id']}_{variant}_{font_label}"
                image_path = images_dir / f"{stem}.png"
                transcript_path = transcripts_dir / f"{stem}.txt"
                if write:
                    img.save(image_path)
                    transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
                    rows.append({
                        "receipt_id": receipt["receipt_id"],
                        "receipt_set": p3b.RECEIPT_SET,
                        "variant": variant,
                        "payload_id": payload_id,
                        "payload_lang": payload_lang,
                        "arm": "B",
                        "font": font_label,
                        "channel": "image",
                        "image_path": str(image_path.relative_to(out_dir)),
                        "transcript_path": str(transcript_path.relative_to(out_dir)),
                        "image_sha256": mr.sha256_file(image_path),
                        "transcript_sha256": mr.sha256_file(transcript_path),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    rows = arm_b_manifest_rows(args.out_dir)
    print(f"Wrote {len(rows)} arm B rendered variants under {args.out_dir}")


if __name__ == "__main__":
    main()
