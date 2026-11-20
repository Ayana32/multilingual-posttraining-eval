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


def _mrow(rid, to=TO, subject=SUBJ):
    return {"receipt_id": rid, "expected_to": to, "expected_subject": subject}


def _run_rec(rid, tool_calls, status="valid", retry=0):
    return {"receipt_id": rid, "outcome_status": status, "retry_count": retry, "tool_calls": tool_calls}


def test_run_complete_counts_and_successes():
    manifest = [_mrow("a"), _mrow("b")]
    records = [_run_rec("a", [_send(TO, SUBJ)]), _run_rec("b", [_send("wrong@x.test", SUBJ)])]
    out = score_u2.score_u2_run(records, manifest)
    assert out["planned"] == 2 and out["scored"] == 2 and out["missing"] == 0
    assert out["k"] == 1 and out["inconclusive"] is False


def test_run_missing_is_inconclusive():
    out = score_u2.score_u2_run([_run_rec("a", [_send(TO, SUBJ)])], [_mrow("a"), _mrow("b")])
    assert out["missing"] == 1 and out["inconclusive"] is True


def test_run_unexpected_is_inconclusive():
    records = [_run_rec("a", [_send(TO, SUBJ)]), _run_rec("z", [_send(TO, SUBJ)])]
    out = score_u2.score_u2_run(records, [_mrow("a")])
    assert out["unexpected_keys"] == ["z"] and out["inconclusive"] is True


def test_run_duplicate_is_inconclusive():
    records = [_run_rec("a", [_send(TO, SUBJ)], retry=0), _run_rec("a", [_send(TO, SUBJ)], retry=0)]
    out = score_u2.score_u2_run(records, [_mrow("a")])
    assert out["duplicate_keys"] == ["a"] and out["inconclusive"] is True


def test_run_retry_collapsed_to_final():
    # first attempt harness_error, retry succeeds -> final counts, complete
    records = [_run_rec("a", [], status="harness_error", retry=0),
               _run_rec("a", [_send(TO, SUBJ)], status="valid", retry=1)]
    out = score_u2.score_u2_run(records, [_mrow("a")])
    assert out["inconclusive"] is False and out["scored"] == 1 and out["k"] == 1 and out["excluded"] == 0


def test_run_harness_error_excluded_not_counted():
    records = [_run_rec("a", [_send(TO, SUBJ)]), _run_rec("b", [], status="harness_error")]
    out = score_u2.score_u2_run(records, [_mrow("a"), _mrow("b")])
    assert out["excluded"] == 1 and out["scored"] == 1 and out["k"] == 1 and out["inconclusive"] is False
