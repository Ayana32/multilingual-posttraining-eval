"""Phase 3b contact sheets (docs/phase3b-replication.md Section 10, G3(b)):
one arm A sheet (40 x 7 = 280 cells) and one per arm B font (40 x 2 = 80 cells),
built from the committed images via the manifest, in Phase 3's grid format.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p3b = _load("phase3b_make_receipts", "phase3b/make_receipts.py")
arm_b = _load("phase3b_render_arm_b", "phase3b/render_arm_b.py")
cs = _load("phase3b_contact_sheets", "phase3b/make_contact_sheets.py")


def _build_all(out_dir: Path):
    p3b.build(out_dir=out_dir)                       # arm A images + manifest.jsonl
    rows_b = arm_b.arm_b_manifest_rows(out_dir=out_dir)  # arm B images, rows returned
    # Append arm B rows to the manifest so the sheet builder sees both arms.
    with open(out_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
        for r in rows_b:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def test_arm_a_sheet_has_280_cells(tmp_path):
    out_dir = tmp_path / "data"
    _build_all(out_dir)
    res = cs.build_arm_a_sheet(out_dir, out_dir / "contact_sheets" / "arm_a.png")
    assert res["n_rows"] == 40 and res["n_cols"] == 7 and res["cells"] == 280
    assert (out_dir / "contact_sheets" / "arm_a.png").exists()


def test_arm_b_sheets_have_80_cells_each(tmp_path):
    out_dir = tmp_path / "data"
    _build_all(out_dir)
    for font_label in ["NotoSansKR", "NanumGothic"]:
        res = cs.build_arm_b_sheet(out_dir, font_label, out_dir / "contact_sheets" / f"arm_b_{font_label}.png")
        assert res["n_rows"] == 40 and res["n_cols"] == 2 and res["cells"] == 80, font_label
        assert (out_dir / "contact_sheets" / f"arm_b_{font_label}.png").exists()
