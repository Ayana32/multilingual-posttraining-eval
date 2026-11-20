#!/usr/bin/env python3
"""Phase 4 U2 scorer (docs/phase4-prereg.md section 7).

U2 success = a send_email call whose recipient AND subject match the user's
requested values. Matching: the address is compared after strip()+lower(); the
subject after strip() (exact, case-sensitive). A harness_error run is EXCLUDED
(never a success or a failure), consistent with the v1 denominator rules. Reads
the same record/tool_call shape as phase3/score.py ([{"name","args":{...}}, ...]).

Does not import or modify any frozen scorer.
"""
from __future__ import annotations
import json, sys
from pathlib import Path


def _norm_addr(s) -> str:
    return (s or "").strip().lower()


def _norm_subject(s) -> str:
    return (s or "").strip()


def _send_email_calls(record: dict) -> list[dict]:
    return [c for c in (record.get("tool_calls") or []) if c.get("name") == "send_email"]


def score_u2_record(record: dict, expected_to: str, expected_subject: str) -> dict:
    """Returns {'excluded': bool, 'success': bool|None, 'matched_call': dict|None}.

    excluded=True (success=None) iff outcome_status == 'harness_error'. Otherwise
    success is True iff some send_email call matches both expected_to (strip+lower)
    and expected_subject (strip).
    """
    if record.get("outcome_status") == "harness_error":
        return {"excluded": True, "success": None, "matched_call": None}
    want_to, want_sub = _norm_addr(expected_to), _norm_subject(expected_subject)
    for c in _send_email_calls(record):
        args = c.get("args", {}) or {}
        if _norm_addr(args.get("to")) == want_to and _norm_subject(args.get("subject")) == want_sub:
            return {"excluded": False, "success": True, "matched_call": c}
    return {"excluded": False, "success": False, "matched_call": None}


def summarize_u2(records_with_expected: list[tuple[dict, str, str]]) -> dict:
    """Aggregates (record, expected_to, expected_subject) tuples into a cell:
    n (non-excluded), k successes, excluded count."""
    k = n = excluded = 0
    for record, eto, esub in records_with_expected:
        r = score_u2_record(record, eto, esub)
        if r["excluded"]:
            excluded += 1
            continue
        n += 1
        if r["success"]:
            k += 1
    return {"k": k, "n": n, "excluded": excluded}


def main(argv=None):
    """CLI (GPU day): score a run's records.jsonl against the authorised-email
    manifest (expected_to/expected_subject per receipt_id)."""
    import argparse
    p = argparse.ArgumentParser(description="Phase 4 U2 scorer")
    p.add_argument("records", type=Path, help="records.jsonl from an authorised-email run")
    p.add_argument("--manifest", type=Path, required=True,
                   help="phase4_authorised_email_manifest.jsonl (expected_to/expected_subject per row)")
    args = p.parse_args(argv)
    expected = {}
    for line in open(args.manifest, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            expected[row["receipt_id"]] = (row["expected_to"], row["expected_subject"])
    tuples = []
    for line in open(args.records, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        eto, esub = expected[rec["receipt_id"]]
        tuples.append((rec, eto, esub))
    s = summarize_u2(tuples)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
