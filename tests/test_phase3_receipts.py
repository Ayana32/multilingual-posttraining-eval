"""Checks on the generated Phase 3 receipts (phase3/make_receipts.py):

1. The fixed seed reproduces identical receipts.jsonl and transcript hashes
   on a fresh run, always (test_same_seed_reproduces_identical_receipts_and_transcript_hashes).
2. The fixed seed also reproduces identical image hashes, but only checked
   when the current Pillow/FreeType versions match the ones recorded in
   phase3/data/environment.json, since PNG bytes can differ across
   Pillow/FreeType/OS versions even for pixel-identical renders; the
   committed images are the canonical input either way (see
   phase3/make_receipts.py's module docstring), this is a secondary check
   (test_same_seed_reproduces_identical_image_hashes_in_matching_environment).
   Both tests regenerate from the same build (see the `regenerated` fixture)
   so the image-hash skip never hides a receipts/transcript failure.
3. Every non-clean transcript contains its payload string exactly.
4. Clean transcripts contain none of the payload strings.
5. No rendered line, in any variant, splits the target email address or
   the quoted subject (see ATOMIC_SPAN_PATTERN in make_receipts.py).

Requires the "phase3" optional dependency group (Pillow); CI installs it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import make_receipts as mr  # noqa: E402

DATA_DIR = REPO_ROOT / "phase3" / "data"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
ENVIRONMENT_PATH = DATA_DIR / "environment.json"
PAYLOADS_PATH = REPO_ROOT / "phase3" / "fixtures" / "payloads.json"


def _load_manifest() -> list[dict]:
    return [json.loads(line) for line in open(MANIFEST_PATH, encoding="utf-8") if line.strip()]


def _load_payloads() -> dict:
    return json.loads(PAYLOADS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """Rebuilds from the fixed seed once, shared by both reproducibility
    tests below, so an image-hash skip never hides (or is hidden by) the
    receipts/transcript result."""
    out_dir = tmp_path_factory.mktemp("phase3_data")
    receipts_path = tmp_path_factory.mktemp("phase3_receipts") / "receipts.jsonl"
    mr.build(out_dir, receipts_path, mr.SEED, mr.N_TUNE, mr.N_EVAL)

    committed = {(r["receipt_id"], r["variant"]): r for r in _load_manifest()}
    fresh = {
        (r["receipt_id"], r["variant"]): r
        for r in (json.loads(line) for line in open(out_dir / "manifest.jsonl", encoding="utf-8") if line.strip())
    }
    return {"out_dir": out_dir, "receipts_path": receipts_path, "committed": committed, "fresh": fresh}


def test_same_seed_reproduces_identical_receipts_and_transcript_hashes(regenerated):
    committed = regenerated["committed"]
    fresh = regenerated["fresh"]
    assert set(committed.keys()) == set(fresh.keys())

    # Plain text, stable across Pillow/FreeType/OS: always compared.
    for key in committed:
        assert committed[key]["transcript_sha256"] == fresh[key]["transcript_sha256"], key

    committed_receipts = (REPO_ROOT / "phase3" / "fixtures" / "receipts.jsonl").read_text(encoding="utf-8")
    fresh_receipts = regenerated["receipts_path"].read_text(encoding="utf-8")
    assert committed_receipts == fresh_receipts


def test_same_seed_reproduces_identical_image_hashes_in_matching_environment(regenerated):
    committed = regenerated["committed"]
    fresh = regenerated["fresh"]

    # Image hashes: only compared when the rendering environment matches,
    # since PNG bytes (not necessarily pixels) can differ across
    # Pillow/FreeType/OS versions. The committed images remain canonical
    # either way; this is a secondary check, not a gate on using them.
    committed_env = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    current_env = mr.environment_info()
    same_rendering_env = (
        committed_env["pillow_version"] == current_env["pillow_version"]
        and committed_env["freetype_version"] == current_env["freetype_version"]
    )
    if not same_rendering_env:
        pytest.skip(
            f"Skipping image-hash comparison: committed environment is Pillow "
            f"{committed_env['pillow_version']}/FreeType {committed_env['freetype_version']}, "
            f"current is Pillow {current_env['pillow_version']}/FreeType {current_env['freetype_version']}. "
            f"receipts.jsonl and transcript hashes are checked by "
            f"test_same_seed_reproduces_identical_receipts_and_transcript_hashes, unconditionally."
        )

    for key in committed:
        assert committed[key]["image_sha256"] == fresh[key]["image_sha256"], key


def _transcript_text_rejoined(transcript_path: Path) -> str:
    """Rejoining the transcript's lines with a single space undoes the
    footer's word-wrap (each wrap point replaced exactly one space with a
    newline; see wrap_text in make_receipts.py), recovering the original
    unwrapped payload string as a contiguous substring."""
    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    return " ".join(lines)


def test_every_non_clean_transcript_contains_its_payload_string_exactly():
    payloads = _load_payloads()
    manifest = _load_manifest()
    non_clean = [r for r in manifest if r["payload_id"] is not None]
    assert non_clean, "expected at least one non-clean manifest row"

    for row in non_clean:
        transcript_path = DATA_DIR / row["transcript_path"]
        rejoined = _transcript_text_rejoined(transcript_path)
        expected = payloads[row["payload_id"]][row["payload_lang"]]
        assert expected in rejoined, (row["receipt_id"], row["variant"])


def test_clean_transcripts_contain_no_payload_string():
    payloads = _load_payloads()
    all_payload_strings = [v for payload in payloads.values() for v in payload.values()]

    manifest = _load_manifest()
    clean_rows = [r for r in manifest if r["variant"] == "clean"]
    assert clean_rows, "expected at least one clean manifest row"

    for row in clean_rows:
        transcript_path = DATA_DIR / row["transcript_path"]
        rejoined = _transcript_text_rejoined(transcript_path)
        for s in all_payload_strings:
            assert s not in rejoined, (row["receipt_id"], s)


def test_no_rendered_line_splits_the_target_address_or_subject():
    """For every non-clean transcript, every atomic span found in the
    source payload text (the target email address and the quoted
    subject, each possibly with an attached Korean particle) must appear
    intact within a single transcript line, never reconstructed only by
    joining two separate lines."""
    payloads = _load_payloads()
    manifest = _load_manifest()
    non_clean = [r for r in manifest if r["payload_id"] is not None]
    assert non_clean, "expected at least one non-clean manifest row"

    checked_spans = 0
    for row in non_clean:
        payload_text = payloads[row["payload_id"]][row["payload_lang"]]
        spans = [m.group(0) for m in mr.ATOMIC_SPAN_PATTERN.finditer(payload_text)]
        if row["payload_id"] == "neutral":
            # The neutral payload is a plain thank-you message; it contains
            # neither the target address nor the subject, so there is
            # nothing to check here.
            assert not spans, (row["receipt_id"], row["variant"], spans)
            continue
        assert spans, (row["receipt_id"], row["variant"], "expected at least one atomic span in the payload text")

        transcript_path = DATA_DIR / row["transcript_path"]
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
        for span in spans:
            checked_spans += 1
            assert any(span in line for line in lines), (
                f"{row['receipt_id']} {row['variant']}: atomic span {span!r} was split across lines: {lines}"
            )
    assert checked_spans > 0
