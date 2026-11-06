"""Checks on phase3/run.py (the run CLI's testable pieces), with a fake
template_fn/generate_fn -- no real model is downloaded or run:

1. select_rows filtering (receipt_set, variants, receipt_ids).
2. run_agent_rows: one record per (row, channel), the harness_error
   retry-once behavior (both the "retry succeeds" and "retry also fails"
   cases), and that the eval guard still applies.
3. Transcription runs: build_transcription_input's structure, and
   run_transcription's guards, field set, and refusal on a clean
   (payload_id is None) row.
4. load_config reads phase3/config.yaml, and its placeholder
   model_revision is refused by validate_model_revision (by design --
   Commit B must pin a real one first).

Requires the "phase3" optional dependency group (Pillow); CI installs it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import harness as hz  # noqa: E402
import run as run_module  # noqa: E402

DATA_DIR = REPO_ROOT / "phase3" / "data"
VALID_REVISION = "a" * 40


def _load_fixtures() -> hz.Fixtures:
    return hz.load_fixtures()


def _load_manifest() -> list[dict]:
    return hz.load_manifest(DATA_DIR)


def _row(receipt_id: str, variant: str) -> dict:
    return next(r for r in _load_manifest() if r["receipt_id"] == receipt_id and r["variant"] == variant)


def _config(**overrides) -> hz.HarnessConfig:
    kwargs = dict(model_id="fake/model", model_revision=VALID_REVISION, max_new_tokens=64)
    kwargs.update(overrides)
    return hz.HarnessConfig(**kwargs)


def _fake_template_fn(messages, tools) -> str:
    parts = []
    for message in messages:
        parts.append(f"<|{message['role']}|>\n")
        for block in message["content"]:
            if block["type"] == "text":
                parts.append(block["text"])
            elif block["type"] == "image":
                parts.append("<image_placeholder>")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# select_rows
# ---------------------------------------------------------------------------

def test_select_rows_filters_by_receipt_set():
    manifest = _load_manifest()
    rows = run_module.select_rows(manifest, "tune")
    assert rows
    assert all(r["receipt_set"] == "tune" for r in rows)
    assert all(r["receipt_id"].startswith("tune_") for r in rows)


def test_select_rows_filters_by_variant_and_receipt_id():
    manifest = _load_manifest()
    rows = run_module.select_rows(manifest, "tune", variants={"clean"}, receipt_ids={"tune_01"})
    assert len(rows) == 1
    assert rows[0]["receipt_id"] == "tune_01"
    assert rows[0]["variant"] == "clean"


def test_select_rows_never_returns_eval_rows_when_tune_requested():
    manifest = _load_manifest()
    rows = run_module.select_rows(manifest, "tune")
    assert all(r["receipt_set"] != "eval" for r in rows)


# ---------------------------------------------------------------------------
# run_agent_rows
# ---------------------------------------------------------------------------

def test_run_agent_rows_produces_one_record_per_row_channel_pair():
    fixtures = _load_fixtures()
    config = _config()
    rows = [_row("tune_01", "clean"), _row("tune_02", "clean")]
    rows_and_channels = [(row, channel) for row in rows for channel in ("image", "text")]

    def generate_fn(rendered_prompt, images):
        return "Nothing unusual.", False

    records = run_module.run_agent_rows(rows_and_channels, fixtures, _fake_template_fn, generate_fn, config, DATA_DIR)

    assert len(records) == 4
    assert {(r["receipt_id"], r["channel"]) for r in records} == {
        ("tune_01", "image"), ("tune_01", "text"), ("tune_02", "image"), ("tune_02", "text"),
    }
    assert all(r["outcome_status"] == "valid" for r in records)


def test_run_agent_rows_retries_once_and_keeps_the_rerun_record_on_success():
    fixtures = _load_fixtures()
    config = _config()
    row = _row("tune_01", "clean")

    call_count = {"n": 0}

    def generate_fn(rendered_prompt, images):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient failure")
        return "All good on the retry.", False

    records = run_module.run_agent_rows([(row, "text")], fixtures, _fake_template_fn, generate_fn, config, DATA_DIR)

    assert call_count["n"] == 2
    assert len(records) == 1
    assert records[0]["outcome_status"] == "valid"
    assert records[0]["final_text"] == "All good on the retry."


def test_run_agent_rows_keeps_the_second_failure_if_retry_also_fails():
    fixtures = _load_fixtures()
    config = _config()
    row = _row("tune_01", "clean")

    call_count = {"n": 0}

    def generate_fn(rendered_prompt, images):
        call_count["n"] += 1
        raise RuntimeError(f"failure #{call_count['n']}")

    records = run_module.run_agent_rows([(row, "text")], fixtures, _fake_template_fn, generate_fn, config, DATA_DIR)

    assert call_count["n"] == 2  # exactly one retry, not more
    assert len(records) == 1
    assert records[0]["outcome_status"] == "harness_error"
    assert "failure #2" in records[0]["raw_output"][-1]


def test_run_agent_rows_refuses_eval_row_before_calling_generate_fn():
    fixtures = _load_fixtures()
    config = _config(commit_b_tag=False)
    row = next(r for r in _load_manifest() if r["receipt_set"] == "eval")

    called = []

    def generate_fn(rendered_prompt, images):
        called.append(1)
        return "should not run", False

    with pytest.raises(hz.EvalGuardError):
        run_module.run_agent_rows([(row, "text")], fixtures, _fake_template_fn, generate_fn, config, DATA_DIR)
    assert called == []


# ---------------------------------------------------------------------------
# Transcription runs
# ---------------------------------------------------------------------------

def test_build_transcription_input_has_only_prompt_and_image():
    fixtures = _load_fixtures()
    row = _row("tune_01", "P1_en")

    model_input = run_module.build_transcription_input(row, fixtures, DATA_DIR)

    assert model_input["tools"] == []
    assert len(model_input["messages"]) == 1
    message = model_input["messages"][0]
    assert message["role"] == "user"
    assert len(message["content"]) == 2
    assert message["content"][0] == {"type": "text", "text": fixtures.transcription_prompt + "\n"}
    assert message["content"][1]["type"] == "image"
    assert isinstance(message["content"][1]["image"], Image.Image)


def test_run_transcription_raises_on_a_clean_row():
    fixtures = _load_fixtures()
    row = _row("tune_01", "clean")
    config = _config()
    with pytest.raises(ValueError):
        run_module.run_transcription(row, fixtures, _fake_template_fn, lambda p, imgs: ("x", False), config, DATA_DIR)


def test_run_transcription_returns_expected_fields():
    fixtures = _load_fixtures()
    row = _row("tune_01", "P1_en")
    config = _config()

    record = run_module.run_transcription(
        row, fixtures, _fake_template_fn, lambda p, imgs: ("transcribed text here", True), config, DATA_DIR
    )

    assert record["receipt_id"] == "tune_01"
    assert record["variant"] == "P1_en"
    assert record["payload_id"] == "P1"
    assert record["payload_lang"] == "en"
    assert record["model_id"] == config.model_id
    assert record["model_revision"] == config.model_revision
    assert record["raw_output"] == "transcribed text here"
    assert record["hit_max_new_tokens"] is True
    assert record["rendered_prompt"] == _fake_template_fn(
        run_module.build_transcription_input(row, fixtures, DATA_DIR)["messages"], [],
    )
    expected_sha256 = hz.prompt_sha256_for_rendered(record["rendered_prompt"], row["image_sha256"])
    assert record["prompt_sha256"] == expected_sha256


def test_run_transcription_refuses_eval_row():
    fixtures = _load_fixtures()
    config = _config(commit_b_tag=False)
    row = next(r for r in _load_manifest() if r["receipt_set"] == "eval" and r["payload_id"] is not None)

    called = []

    def generate_fn(rendered_prompt, images):
        called.append(1)
        return "should not run", False

    with pytest.raises(hz.EvalGuardError):
        run_module.run_transcription(row, fixtures, _fake_template_fn, generate_fn, config, DATA_DIR)
    assert called == []


def test_run_transcription_refuses_bad_model_revision():
    fixtures = _load_fixtures()
    row = _row("tune_01", "P1_en")
    config = _config(model_revision="main")
    with pytest.raises(hz.InvalidRevisionError):
        run_module.run_transcription(row, fixtures, _fake_template_fn, lambda p, imgs: ("x", False), config, DATA_DIR)


def test_run_transcription_rows_runs_only_given_rows():
    fixtures = _load_fixtures()
    config = _config()
    rows = [_row("tune_01", "P1_en"), _row("tune_01", "neutral_ko")]

    records = run_module.run_transcription_rows(
        rows, fixtures, _fake_template_fn, lambda p, imgs: ("x", False), config, DATA_DIR
    )

    assert len(records) == 2
    assert {r["variant"] for r in records} == {"P1_en", "neutral_ko"}


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_reads_phase3_config_yaml():
    config = run_module.load_config()
    assert config.model_id
    assert config.max_new_tokens > 0
    assert config.commit_b_tag is False

    # The committed config.yaml deliberately ships a placeholder revision,
    # refused by design until Commit B pins a real commit hash.
    with pytest.raises(hz.InvalidRevisionError):
        hz.validate_model_revision(config.model_revision)
