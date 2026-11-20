"""The Phase 4 eval run path must use the committed PNGs and must NOT re-render
(prereg: frozen eval set). Confirms (a) every image row's committed PNG matches
its manifest sha256, (b) build_model_input hands back exactly the committed file
bytes (it loaded them), and (c) the harness run path contains no renderer, so a
run cannot re-render."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path

import harness as hz
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "phase4" / "data" / "eval"
MANIFEST = REPO / "runs" / "phase4_manifest.jsonl"
FIX = hz.load_fixtures()


def _rows():
    return [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]


def test_committed_pngs_match_manifest_sha():
    img_rows = [r for r in _rows() if r.get("image_path")]
    assert len(img_rows) == 60  # 20 receipts x 3 variants
    for r in img_rows:
        p = EVAL_DIR / r["image_path"]
        assert p.exists(), f"missing committed PNG {p}"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["image_sha256"], \
            f"committed PNG differs from manifest sha for {r['receipt_id']} {r['variant']}"


def test_run_path_loads_committed_bytes():
    row = next(r for r in _rows() if r.get("image_path"))
    built = hz.build_model_input(row, "image", FIX, EVAL_DIR)
    img_block = built["messages"][1]["content"][2]
    assert img_block["type"] == "image"
    committed = (EVAL_DIR / row["image_path"]).read_bytes()
    expected = Image.open(io.BytesIO(committed)).convert("RGB")
    assert img_block["image"].tobytes() == expected.tobytes()


def test_harness_run_path_has_no_renderer():
    src = (REPO / "phase3" / "harness.py").read_text(encoding="utf-8")
    assert "import make_receipts" not in src
    assert "render_receipt" not in src   # the run path reads committed bytes via load_image, never renders
