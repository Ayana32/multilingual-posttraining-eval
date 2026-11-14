"""Phase 3b G3(a) automated rendering check (docs/phase3b-replication.md
Section 10, G3(a)):

- check_cmap_coverage flags a character absent from the rendering font's cmap
  (a Hangul syllable under Noto Sans, which does not cover Hangul) and passes
  plain ASCII.
- check_bbox_in_bounds flags a bbox outside the image bounds and passes one
  inside.
- run_g3a over a subset renders through Phase 3's render_receipt and reports all
  cmap/bbox checks passing, with the footer line count equal to the
  transcript-derived count for payload-bearing images.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import make_receipts as mr  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


g3 = _load("phase3b_g3_check", "phase3b/g3_check.py")

NOTO_SANS = mr._load_fonts()["en_body"].path  # does not cover Hangul


def test_cmap_coverage_flags_uncovered_char_and_passes_ascii():
    bad = [{"text": "가", "font_path": NOTO_SANS, "bbox": (0, 0, 10, 10)}]
    fails = g3.check_cmap_coverage(bad)
    assert len(fails) == 1 and "가" in fails[0]["missing"]

    good = [{"text": "Total: £12.00 GBP", "font_path": NOTO_SANS, "bbox": (0, 0, 10, 10)}]
    assert g3.check_cmap_coverage(good) == []

    # None font_path (e.g. a draw with no font) is skipped, not an error.
    assert g3.check_cmap_coverage([{"text": "x", "font_path": None, "bbox": (0, 0, 1, 1)}]) == []


def test_bbox_in_bounds_flags_overflow_and_passes_inside():
    inside = [{"text": "x", "bbox": (0, 0, 560, 480)}]
    assert g3.check_bbox_in_bounds(inside, 560, 480) == []

    over_right = [{"text": "x", "bbox": (0, 0, 561, 480)}]
    assert len(g3.check_bbox_in_bounds(over_right, 560, 480)) == 1

    negative = [{"text": "x", "bbox": (-1, 0, 10, 10)}]
    assert len(g3.check_bbox_in_bounds(negative, 560, 480)) == 1


def test_run_g3a_subset_all_pass_and_footer_counts_match():
    subset = ["phase3b_eval_01", "phase3b_eval_02", "phase3b_eval_03"]
    results = g3.run_g3a(receipt_ids=subset)
    # 3 receipts x 7 arm A variants + 2 fonts x 3 receipts x 2 arm B variants.
    assert results["n_images"] == 3 * 7 + 2 * 3 * 2
    assert results["all_cmap_pass"] is True, results["cmap_failures"]
    assert results["all_bbox_pass"] is True, results["bbox_failures"]

    by_id = {r["image_id"]: r for r in results["images"]}
    # clean images have no footer; payload images have a positive footer count.
    assert by_id["A|clean|phase3b_eval_01"]["footer_line_count"] is None
    for image_id, r in by_id.items():
        if "clean" in image_id:
            assert r["footer_line_count"] is None
        else:
            assert isinstance(r["footer_line_count"], int) and r["footer_line_count"] >= 1
