"""Unit tests for phase4/run_validation.py (shared by U2 and the main eval)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
import run_validation as rv  # noqa: E402

KEY = lambda r: r["receipt_id"]


def rec(rid, status="valid", retry=0):
    return {"receipt_id": rid, "outcome_status": status, "retry_count": retry, "tool_calls": []}


def test_complete():
    v = rv.validate_runs([rec("a"), rec("b")], ["a", "b"], KEY)
    assert v["complete"] and v["planned"] == 2 and v["scored"] == 2 and v["missing"] == 0 and v["excluded"] == 0


def test_missing():
    v = rv.validate_runs([rec("a")], ["a", "b"], KEY)
    assert not v["complete"] and v["missing"] == 1 and v["missing_keys"] == ["b"]


def test_unexpected():
    v = rv.validate_runs([rec("a"), rec("z")], ["a"], KEY)
    assert not v["complete"] and v["unexpected_keys"] == ["z"]


def test_duplicate_same_retry_count_flagged():
    v = rv.validate_runs([rec("a", retry=0), rec("a", retry=0)], ["a"], KEY)
    assert v["duplicate_keys"] == ["a"] and not v["complete"]


def test_retry_collapsed_to_final_not_duplicate():
    first = rec("a", status="harness_error", retry=0)
    final = rec("a", status="valid", retry=1)
    v = rv.validate_runs([first, final], ["a"], KEY)
    assert v["duplicate_keys"] == [] and v["complete"]
    assert v["final"]["a"] is final and v["scored"] == 1 and v["excluded"] == 0


def test_harness_error_final_excluded():
    v = rv.validate_runs([rec("a", status="harness_error")], ["a"], KEY)
    assert v["excluded"] == 1 and v["scored"] == 0 and v["excluded_keys"] == ["a"]
