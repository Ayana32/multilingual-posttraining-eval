#!/usr/bin/env python3
"""Phase 3b G3 part (a): the automated rendering check over all 440 images
(docs/phase3b-replication.md Section 10, G3(a)). No manual review; parts (b)
and (c) are the human contact-sheet and line-by-line passes done separately.

Three mechanical checks per image:
  1. cmap coverage: every character of every rendered string is present in the
     cmap of the font that rendered it (catches missing-glyph/"tofu").
  2. bbox in bounds: every text bounding box (draw.textbbox) lies entirely
     inside the image's pixel bounds (catches clipped/overflowing text).
  3. footer line count: for every image with a footer, the mechanical count
     phase3/failure_cases.py::n_transcript_lines_for_payload (reused unchanged).

How the per-string font is known WITHOUT re-deriving the layout: each image is
re-rendered through Phase 3's own render_receipt while the ImageDraw.text calls
are traced, so the checks see exactly the (text, font, position, bbox) that were
actually drawn. Nothing is re-implemented from render_receipt's internals.

The result (pass/fail per image per check) is returned as a dict and, in the
final step, written to a committed file under report/phase3b/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

from PIL import ImageDraw
from fontTools.ttLib import TTFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import make_receipts as mr  # noqa: E402  (render_receipt, reused unchanged)
import failure_cases as fc  # noqa: E402  (n_transcript_lines_for_payload)

FIXTURES_DIR = REPO_ROOT / "phase3" / "fixtures"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p3b = _load("phase3b_make_receipts", "phase3b/make_receipts.py")
arm_b = _load("phase3b_render_arm_b", "phase3b/render_arm_b.py")


@lru_cache(maxsize=None)
def _cmap_for(path: str) -> frozenset:
    return frozenset(TTFont(path).getBestCmap().keys())


def trace_render(render_fn):
    """Runs render_fn() (which must draw via a PIL ImageDraw and return
    (image, transcript_lines)) while recording every draw.text call as
    {text, xy, font_path, bbox}. Returns (image, transcript_lines, draws)."""
    draws: list[dict] = []
    orig_text = ImageDraw.ImageDraw.text

    def traced_text(self, xy, text, *args, **kwargs):
        font = kwargs.get("font")
        if font is None and args:
            font = args[0]
        bbox = self.textbbox(xy, text, font=font)
        draws.append({
            "text": text,
            "xy": tuple(xy),
            "font_path": getattr(font, "path", None),
            "bbox": tuple(bbox),
        })
        return orig_text(self, xy, text, *args, **kwargs)

    ImageDraw.ImageDraw.text = traced_text
    try:
        image, transcript_lines = render_fn()
    finally:
        ImageDraw.ImageDraw.text = orig_text
    return image, transcript_lines, draws


def check_cmap_coverage(draws: list[dict]) -> list[dict]:
    """Returns a list of failures: {text, font_path, missing} for any drawn
    string containing a character absent from its font's cmap."""
    failures = []
    for d in draws:
        if d["font_path"] is None:
            continue
        cmap = _cmap_for(d["font_path"])
        missing = sorted({ch for ch in d["text"] if ord(ch) not in cmap})
        if missing:
            failures.append({"text": d["text"], "font_path": d["font_path"], "missing": missing})
    return failures


def check_bbox_in_bounds(draws: list[dict], width: int, height: int) -> list[dict]:
    """Returns a list of failures for any text bbox not entirely inside
    (0, 0, width, height)."""
    failures = []
    for d in draws:
        left, top, right, bottom = d["bbox"]
        if left < 0 or top < 0 or right > width or bottom > height:
            failures.append({"text": d["text"], "bbox": d["bbox"], "bounds": (width, height)})
    return failures


def _payload_text(payloads: dict, payload_id: str | None, payload_lang: str | None) -> str | None:
    if payload_id is None or payload_lang is None:
        return None
    return payloads[payload_id][payload_lang]


def g3a_for_render(image_id: str, render_fn, payload_text: str | None) -> dict:
    image, transcript_lines, draws = trace_render(render_fn)
    w, h = image.size
    cmap_failures = check_cmap_coverage(draws)
    bbox_failures = check_bbox_in_bounds(draws, w, h)
    footer_lines = None
    if payload_text is not None:
        transcript_text = "\n".join(transcript_lines) + "\n"
        footer_lines = fc.n_transcript_lines_for_payload(transcript_text, payload_text)
    return {
        "image_id": image_id,
        "width": w, "height": h,
        "n_draws": len(draws),
        "cmap_pass": not cmap_failures,
        "cmap_failures": cmap_failures,
        "bbox_pass": not bbox_failures,
        "bbox_failures": bbox_failures,
        "footer_line_count": footer_lines,
    }


def run_g3a(seed: int = p3b.SEED, n_eval: int = p3b.N_EVAL, receipt_ids: list[str] | None = None) -> dict:
    """Runs G3(a) over all arm A (280) and arm B (160) images = 440. Returns the
    full per-image results plus a summary. receipt_ids restricts to a subset
    (used by the test for speed); None means all 40."""
    payloads = json.loads((FIXTURES_DIR / "payloads.json").read_text(encoding="utf-8"))
    receipts = p3b.generate_receipts(seed, n_eval)
    if receipt_ids is not None:
        receipts = [r for r in receipts if r["receipt_id"] in set(receipt_ids)]
    fonts_a = mr._load_fonts()
    fonts_b = arm_b.load_arm_b_fonts()

    results: list[dict] = []

    # Arm A: 7 variants per receipt.
    for receipt in receipts:
        for variant, payload_id, payload_lang in mr.VARIANTS:
            image_id = f"A|{variant}|{receipt['receipt_id']}"
            pt = _payload_text(payloads, payload_id, payload_lang)
            results.append(g3a_for_render(
                image_id,
                lambda r=receipt, v=variant, pid=payload_id, pl=payload_lang:
                    mr.render_receipt(r, v, pid, pl, payloads, fonts_a),
                pt,
            ))

    # Arm B: P1_en, P1_ko x 2 fonts.
    for font_label in arm_b.ARM_B_FONTS:
        fonts = fonts_b[font_label]
        for receipt in receipts:
            for variant, payload_id, payload_lang in arm_b.ARM_B_VARIANTS:
                image_id = f"B|{font_label}|{variant}|{receipt['receipt_id']}"
                pt = _payload_text(payloads, payload_id, payload_lang)
                results.append(g3a_for_render(
                    image_id,
                    lambda r=receipt, v=variant, pid=payload_id, pl=payload_lang, fd=fonts:
                        mr.render_receipt(r, v, pid, pl, payloads, fd),
                    pt,
                ))

    n_images = len(results)
    cmap_fail = [r["image_id"] for r in results if not r["cmap_pass"]]
    bbox_fail = [r["image_id"] for r in results if not r["bbox_pass"]]
    return {
        "check": "G3(a) automated rendering check",
        "n_images": n_images,
        "all_cmap_pass": not cmap_fail,
        "all_bbox_pass": not bbox_fail,
        "cmap_failures": cmap_fail,
        "bbox_failures": bbox_fail,
        "images": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "report" / "phase3b" / "g3a_results.json")
    args = parser.parse_args()
    results = run_g3a()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"G3(a): {results['n_images']} images; "
          f"cmap {'PASS' if results['all_cmap_pass'] else 'FAIL'}, "
          f"bbox {'PASS' if results['all_bbox_pass'] else 'FAIL'} -> {args.out}")


if __name__ == "__main__":
    main()
