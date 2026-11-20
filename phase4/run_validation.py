#!/usr/bin/env python3
"""Shared run/manifest validation for Phase 4 scoring (prereg sections 6, 7).

One function used by BOTH phase4/score_u2.py (authorised-email eval) and
phase4/score_main.py (H1/U1 main eval): it checks that every planned manifest
row is present exactly once, collapses retries to the final record per row (the
v1 retry rule: a harness_error run is re-run once, so the final attempt -- the
highest retry_count -- is the record that counts), and reports planned / scored
/ missing / excluded counts plus duplicate and unexpected keys and a
`complete` flag. Does not import or change any frozen scorer.
"""
from __future__ import annotations
from collections import defaultdict


def collapse_retries(records, key_fn):
    """Final record per key by the v1 retry rule: keep the highest retry_count
    (ties broken by file order -> last). Returns (final: dict[key]->record,
    duplicate_keys: list) where a duplicate key has >1 record sharing the max
    retry_count (a true duplicate, not a retry chain)."""
    by_key = defaultdict(list)
    for r in records:
        by_key[key_fn(r)].append(r)
    final, duplicates = {}, []
    for key, recs in by_key.items():
        if len(recs) == 1:
            final[key] = recs[0]
            continue
        maxrc = max(r.get("retry_count", 0) for r in recs)
        top = [r for r in recs if r.get("retry_count", 0) == maxrc]
        if len(top) > 1:
            duplicates.append(key)
        final[key] = top[-1]
    return final, duplicates


def validate_runs(records, planned_keys, key_fn):
    """Validate a run's records against the planned manifest keys.

    Returns a dict with: planned, scored, missing, excluded (counts), plus
    missing_keys / unexpected_keys / duplicate_keys lists, the collapsed
    `final` map, and `complete` (True iff no missing, unexpected or duplicate
    keys). `scored` counts planned keys present with a non-harness_error final
    record; `excluded` counts planned keys whose final record is harness_error.
    """
    planned = set(planned_keys)
    final, duplicate_keys = collapse_retries(records, key_fn)
    present = set(final)
    missing = planned - present
    unexpected = present - planned
    excluded = [k for k in (planned & present)
                if final[k].get("outcome_status") == "harness_error"]
    scored = (planned & present) - set(excluded)
    complete = (not missing) and (not unexpected) and (not duplicate_keys)
    return {
        "planned": len(planned),
        "scored": len(scored),
        "missing": len(missing),
        "excluded": len(excluded),
        "missing_keys": sorted(map(str, missing)),
        "unexpected_keys": sorted(map(str, unexpected)),
        "duplicate_keys": sorted(map(str, duplicate_keys)),
        "excluded_keys": sorted(map(str, excluded)),
        "final": final,
        "complete": complete,
    }
