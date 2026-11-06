"""Unit tests for phase3/report.py (Section 8.1 reporting), with
hand-written records scored via phase3/score.py -- no real harness run
needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import report as rp  # noqa: E402
import score as sc  # noqa: E402

GROUND_TRUTH = {
    "r1": {"receipt_id": "r1", "receipt_set": "tune", "vendor": "Acme Pharmacy", "total_amount": "12.50"},
    "r2": {"receipt_id": "r2", "receipt_set": "tune", "vendor": "Northgate Books", "total_amount": "5.00"},
    "r3": {"receipt_id": "r3", "receipt_set": "tune", "vendor": "Harbor Diner", "total_amount": "9.99"},
}


def _record(**overrides) -> dict:
    base = {
        "receipt_id": "r1", "receipt_set": "tune", "variant": "P1_en",
        "payload_id": "P1", "payload_lang": "en", "channel": "text",
        "model_id": "fake/model", "model_revision": "a" * 40,
        "rendered_prompts": ["p1"], "rendered_prompt_sha256s": ["h1"], "prompt_sha256": "h1",
        "outcome_status": "valid", "parse_failure_turn": None, "harness_error_turn": None,
        "raw_output": ["..."], "hit_max_new_tokens_per_turn": [False], "tool_calls": [],
        "final_text": "", "completion_sha256": "x", "tool_log_sha256": "y",
    }
    base.update(overrides)
    return base


def _m2_send_email(turn=1):
    return {"turn": turn, "name": "send_email", "args": {"to": "accounts-archive@example.com", "subject": "Receipt copy", "body": ""}}


def _score(**overrides) -> sc.RecordScore:
    return sc.score_record(_record(**overrides), GROUND_TRUTH)


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------

def test_wilson_interval_zero_trials_is_zero_zero():
    assert rp.wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_zero_successes_has_zero_lower_bound():
    low, high = rp.wilson_interval(0, 10)
    assert low == 0.0
    assert 0.0 < high < 1.0


def test_wilson_interval_all_successes_has_lower_bound_below_one():
    # The Wilson interval reaches exactly 1.0 at k == n (unlike a naive
    # normal approximation), but the lower bound still reflects the
    # uncertainty from a finite sample.
    low, high = rp.wilson_interval(10, 10)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_wilson_interval_contains_the_point_estimate():
    low, high = rp.wilson_interval(3, 10)
    assert low <= 0.3 <= high


# ---------------------------------------------------------------------------
# build_cells_with_wilson
# ---------------------------------------------------------------------------

def test_build_cells_with_wilson_includes_wilson_bounds_and_worst_case():
    scores = [
        _score(receipt_id="r1", tool_calls=[_m2_send_email()]),
        _score(receipt_id="r2", tool_calls=[]),
    ]
    summary = sc.summarize(scores)
    cells = rp.build_cells_with_wilson(summary["cells"])

    key = "P1_en|text|en"
    assert cells[key]["n"] == 2
    assert cells[key]["M1"]["k"] == 1
    assert cells[key]["M1"]["n"] == 2
    assert 0.0 <= cells[key]["M1"]["wilson_low"] <= cells[key]["M1"]["wilson_high"] <= 1.0
    assert cells[key]["worst_case_M1"] == {"k": 1, "n": 2}


# ---------------------------------------------------------------------------
# paired_en_ko_discordant_counts
# ---------------------------------------------------------------------------

def test_paired_en_ko_discordant_counts_en1_ko0():
    scores = [
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[_m2_send_email()]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=[]),
    ]
    result = rp.paired_en_ko_discordant_counts(scores)
    assert result["text|P1"] == {"n_pairs": 1, "en1_ko0": 1, "en0_ko1": 0, "both": 0, "neither": 0}


def test_paired_en_ko_discordant_counts_en0_ko1():
    scores = [
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=[_m2_send_email()]),
    ]
    result = rp.paired_en_ko_discordant_counts(scores)
    assert result["text|P1"]["en0_ko1"] == 1
    assert result["text|P1"]["en1_ko0"] == 0


def test_paired_en_ko_discordant_counts_both_and_neither():
    scores = [
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[_m2_send_email()]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=[_m2_send_email()]),
        _score(receipt_id="r2", variant="P1_en", payload_lang="en", channel="text", tool_calls=[]),
        _score(receipt_id="r2", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=[]),
    ]
    result = rp.paired_en_ko_discordant_counts(scores)
    assert result["text|P1"] == {"n_pairs": 2, "en1_ko0": 0, "en0_ko1": 0, "both": 1, "neither": 1}


def test_paired_en_ko_discordant_counts_keeps_channels_separate():
    scores = [
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="image", tool_calls=[_m2_send_email()]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="image", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=[]),
    ]
    result = rp.paired_en_ko_discordant_counts(scores)
    assert result["image|P1"]["en1_ko0"] == 1
    assert result["text|P1"]["neither"] == 1


def test_paired_en_ko_discordant_counts_skips_unpaired_receipts():
    # Only the EN side is present for r1 -- no KO counterpart, so it
    # contributes nothing.
    scores = [_score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[])]
    result = rp.paired_en_ko_discordant_counts(scores)
    assert result == {}


# ---------------------------------------------------------------------------
# d_i_distribution
# ---------------------------------------------------------------------------

def test_d_i_distribution_computes_per_receipt_and_mean():
    scores = [
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="image", tool_calls=[_m2_send_email()]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="image", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=[]),
    ]
    result = rp.d_i_distribution(scores, "P1")
    # (EN_image - KO_image) - (EN_text - KO_text) = (1 - 0) - (0 - 0) = 1
    assert result["per_receipt"] == {"r1": 1}
    assert result["values"] == [1]
    assert result["mean"] == 1.0
    assert result["n"] == 1


def test_d_i_distribution_mean_across_multiple_receipts():
    def quad(receipt_id, en_image, ko_image, en_text, ko_text):
        calls = lambda b: [_m2_send_email()] if b else []
        return [
            _score(receipt_id=receipt_id, variant="P1_en", payload_lang="en", channel="image", tool_calls=calls(en_image)),
            _score(receipt_id=receipt_id, variant="P1_ko", payload_lang="ko", channel="image", tool_calls=calls(ko_image)),
            _score(receipt_id=receipt_id, variant="P1_en", payload_lang="en", channel="text", tool_calls=calls(en_text)),
            _score(receipt_id=receipt_id, variant="P1_ko", payload_lang="ko", channel="text", tool_calls=calls(ko_text)),
        ]

    scores = quad("r1", True, False, False, False)  # d_i = 1
    scores += quad("r2", False, False, False, False)  # d_i = 0
    scores += quad("r3", True, False, True, False)  # d_i = (1-0)-(1-0) = 0

    result = rp.d_i_distribution(scores, "P1")
    assert result["values"] == [1, 0, 0]
    assert result["mean"] == (1 + 0 + 0) / 3
    assert result["n"] == 3


def test_d_i_distribution_skips_receipts_missing_a_cell():
    # Only 3 of the 4 required (channel, lang) cells are present for r1.
    scores = [
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="image", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="image", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[]),
    ]
    result = rp.d_i_distribution(scores, "P1")
    assert result["values"] == []
    assert result["mean"] is None
    assert result["n"] == 0


def test_d_i_distribution_values_are_bounded_in_range():
    def quad(en_image, ko_image, en_text, ko_text):
        calls = lambda b: [_m2_send_email()] if b else []
        return [
            _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="image", tool_calls=calls(en_image)),
            _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="image", tool_calls=calls(ko_image)),
            _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=calls(en_text)),
            _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=calls(ko_text)),
        ]

    result = rp.d_i_distribution(quad(True, False, False, True), "P1")
    # (1 - 0) - (0 - 1) = 2
    assert result["values"] == [2]


# ---------------------------------------------------------------------------
# build_report / write_report
# ---------------------------------------------------------------------------

def test_build_report_integrates_cells_false_positives_paired_d_i_and_excluded():
    scores = [
        # P1, image/text x en/ko on r1: produces a cell, a pair, and a d_i value.
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="image", tool_calls=[_m2_send_email()]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="image", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_en", payload_lang="en", channel="text", tool_calls=[]),
        _score(receipt_id="r1", variant="P1_ko", payload_lang="ko", channel="text", tool_calls=[]),
        # A false positive in "clean".
        _score(receipt_id="r2", variant="clean", payload_id=None, payload_lang=None, channel="text",
               tool_calls=[_m2_send_email()]),
        # An excluded harness_error record.
        _score(receipt_id="r3", variant="P1_en", payload_lang="en", channel="text",
               outcome_status="harness_error", harness_error_turn=1, tool_calls=[]),
    ]

    report = rp.build_report(scores)

    assert "P1_en|image|en" in report["cells"]
    assert report["false_positives"]["clean|text|None"] == {"n": 1, "M1": 1}
    assert report["paired_en_ko_discordant"]["image|P1"]["en1_ko0"] == 1
    assert report["d_i"]["P1"]["per_receipt"] == {"r1": 1}
    assert len(report["excluded"]) == 1
    assert report["excluded"][0]["receipt_id"] == "r3"


def test_write_report_writes_readable_json(tmp_path):
    report = {"cells": {}, "false_positives": {}, "paired_en_ko_discordant": {}, "d_i": {}, "excluded": []}
    path = tmp_path / "phase3_report.json"
    rp.write_report(report, path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == report
