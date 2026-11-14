"""Phase 3b arm B rendering checks (docs/phase3b-replication.md Section 7):

1. The committed NanumGothic-Regular.ttf matches the pre-registered SHA-256,
   covers Latin and the Hangul syllable range bounds, and is a STATIC font
   (no fvar) -- so set_variation_by_name must not be called on it. Reads the
   committed file only; no network (CI runs with `-m "not network"`).
2. load_arm_b_fonts wires a single font for BOTH scripts per candidate
   (en_footer is ko_footer), and NotoSansKR reuses Phase 3's own ko_footer.
3. Determinism check: P1_ko under the NotoSansKR candidate is byte-identical to
   arm A's P1_ko (same font, same size, same text, Phase 3's own render_receipt)
   for all 40 receipts, while P1_en under NotoSansKR differs from arm A's P1_en
   (a different footer font). Image-byte equality is only asserted when the
   Pillow/FreeType versions match arm A's render environment; the transcript
   (text) equality is asserted unconditionally.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "phase3" / "fonts"
NANUM = FONTS_DIR / "NanumGothic-Regular.ttf"
NANUM_SHA256 = "76f45ef4a6bcff344c837c95a7dcc26e017e38b5846d5ae0cdcb5b86be2e2d31"

sys.path.insert(0, str(REPO_ROOT / "phase3"))
import make_receipts as mr  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p3b = _load("phase3b_make_receipts", "phase3b/make_receipts.py")
arm_b = _load("phase3b_render_arm_b", "phase3b/render_arm_b.py")


def test_committed_nanum_font_sha_coverage_and_static():
    assert NANUM.exists(), "NanumGothic-Regular.ttf must be committed in phase3/fonts/"
    actual = hashlib.sha256(NANUM.read_bytes()).hexdigest()
    assert actual == NANUM_SHA256
    f = TTFont(str(NANUM))
    cmap = f.getBestCmap()
    assert 0x41 in cmap and 0x61 in cmap, "must cover Latin A/a"
    assert 0xAC00 in cmap and 0xD7A3 in cmap, "must cover Hangul syllable bounds"
    assert "fvar" not in f, "NanumGothic-Regular.ttf must be static (no fvar)"


def test_arm_b_fonts_use_one_font_for_both_scripts():
    fonts = arm_b.load_arm_b_fonts()
    assert set(fonts) == {"NotoSansKR", "NanumGothic"}
    for label, fd in fonts.items():
        # Single footer font for both en and ko (same object).
        assert fd["en_footer"] is fd["ko_footer"], label
    # NotoSansKR reuses Phase 3's own ko_footer object (what makes P1_ko exact).
    base = mr._load_fonts()
    # Same underlying font file and size as Phase 3's ko_footer.
    assert fonts["NotoSansKR"]["ko_footer"].path == base["ko_footer"].path
    assert fonts["NotoSansKR"]["ko_footer"].size == base["ko_footer"].size


@pytest.fixture(scope="module")
def arm_a_and_b(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("p3b_data")
    p3b.build(out_dir=out_dir)                 # arm A images/transcripts
    arm_b.arm_b_manifest_rows(out_dir=out_dir)  # arm B images/transcripts, same dir
    return out_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env_matches(out_dir: Path) -> bool:
    env = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))
    cur = mr.environment_info()
    return (env["pillow_version"] == cur["pillow_version"]
            and env["freetype_version"] == cur["freetype_version"])


def test_p1_ko_notosanskr_is_identical_to_arm_a(arm_a_and_b):
    out_dir = arm_a_and_b
    img_dir = out_dir / "images"
    tr_dir = out_dir / "transcripts"
    env_ok = _env_matches(out_dir)
    checked = 0
    for i in range(1, 41):
        rid = f"phase3b_eval_{i:02d}"
        arm_a_img = img_dir / f"{rid}_P1_ko.png"
        arm_b_img = img_dir / f"{rid}_P1_ko_NotoSansKR.png"
        arm_a_tr = tr_dir / f"{rid}_P1_ko.txt"
        arm_b_tr = tr_dir / f"{rid}_P1_ko_NotoSansKR.txt"
        # Transcript (text) must match unconditionally.
        assert arm_a_tr.read_text() == arm_b_tr.read_text(), rid
        if env_ok:
            assert _sha(arm_a_img) == _sha(arm_b_img), rid
        checked += 1
    assert checked == 40
    if not env_ok:
        pytest.skip("Pillow/FreeType mismatch: image-byte identity not asserted; transcripts checked.")


def test_p1_en_notosanskr_differs_from_arm_a(arm_a_and_b):
    """The candidate changes P1_en's footer font (Noto Sans -> Noto Sans KR), so
    at least one of the 40 P1_en images must differ from arm A's."""
    out_dir = arm_a_and_b
    img_dir = out_dir / "images"
    differ = 0
    for i in range(1, 41):
        rid = f"phase3b_eval_{i:02d}"
        a = img_dir / f"{rid}_P1_en.png"
        b = img_dir / f"{rid}_P1_en_NotoSansKR.png"
        if _sha(a) != _sha(b):
            differ += 1
    assert differ > 0, "P1_en under NotoSansKR should differ from arm A's Noto Sans P1_en"
