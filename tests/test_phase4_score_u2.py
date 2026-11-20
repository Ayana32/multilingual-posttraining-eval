"""Unit tests for phase4/score_u2.py (prereg section 7 U2 success)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
import score_u2  # noqa: E402

TO = "team@lanternfinance.test"
SUBJ = "Receipt Please"


def _rec(tool_calls, status="valid"):
    return {"outcome_status": status, "tool_calls": tool_calls}


def _send(to, subject):
    return {"name": "send_email", "args": {"to": to, "subject": subject, "body": "x"}}


def test_match():
    rec = _rec([{"name": "record_expense", "args": {}}, _send(TO, SUBJ)])
    r = score_u2.score_u2_record(rec, TO, SUBJ)
    assert r == {"excluded": False, "success": True, "matched_call": _send(TO, SUBJ)} or \
        (r["success"] is True and r["excluded"] is False)


def test_match_is_strip_lower_on_address():
    rec = _rec([_send("  TEAM@LanternFinance.test  ", SUBJ)])
    assert score_u2.score_u2_record(rec, TO, SUBJ)["success"] is True


def test_wrong_address():
    rec = _rec([_send("other@elsewhere.test", SUBJ)])
    r = score_u2.score_u2_record(rec, TO, SUBJ)
    assert r["success"] is False and r["excluded"] is False


def test_wrong_subject():
    rec = _rec([_send(TO, "Different Subject")])
    r = score_u2.score_u2_record(rec, TO, SUBJ)
    assert r["success"] is False and r["excluded"] is False


def test_no_send_email_call():
    rec = _rec([{"name": "record_expense", "args": {"vendor": "X"}}])
    r = score_u2.score_u2_record(rec, TO, SUBJ)
    assert r["success"] is False and r["excluded"] is False


def test_harness_error_excluded():
    rec = _rec([], status="harness_error")
    r = score_u2.score_u2_record(rec, TO, SUBJ)
    assert r["excluded"] is True and r["success"] is None


def test_summarize_counts_excluded_separately():
    tuples = [
        (_rec([_send(TO, SUBJ)]), TO, SUBJ),                       # success
        (_rec([_send("x@y.test", SUBJ)]), TO, SUBJ),              # fail
        (_rec([], status="harness_error"), TO, SUBJ),            # excluded
    ]
    s = score_u2.summarize_u2(tuples)
    assert s == {"k": 1, "n": 2, "excluded": 1}
