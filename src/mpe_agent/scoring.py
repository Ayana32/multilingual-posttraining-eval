"""v1 scoring and reporting (docs/v1-spec.md Sections 2-3).

This module does NOT define any measure. M1/M2/M3/TASK/CO, worst-case M1
and the Wilson interval are reused verbatim from the frozen Phase 3 code
(`phase3/score.py`, `phase3/report.py`), imported the same way the Phase 3
tests and `mpe_agent.openai_adapter` import them -- there is no `phase3`
package to install, so the directory is added to `sys.path`. Reusing the
exact scorer is what makes the G1 regression (re-scoring the stored Phase
3/3b records) meaningful: a divergence here would be a wiring bug in v1,
not a scoring change.

What this module adds on top of the reused scorer:
- `score_jsonl` / `score_records`: load records and score them.
- `cells_with_wilson`: per (variant, channel, payload_lang) k/n with a
  Wilson 95% interval for every rate (reuses `report.build_cells_with_wilson`).
- `run_status_counts`: valid / parse_failure / harness_error counts and
  the recovered-call count, against a planned-run total.
- `paired_change`: per-receipt direction of change of one measure between
  two conditions (e.g. image vs text), with the number of pairs dropped.

The S measure's rule-based labelling is intentionally NOT here: its rules
are frozen separately (see `mpe_agent.s_report` for the no-final-text
branch; the refusal/sent/no-claim classifier is pending sign-off before
it is written).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Iterable

# phase3/score.py and phase3/report.py are imported, never modified, via the
# same sys.path shim mpe_agent.openai_adapter uses (no phase3 package exists).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHASE3_DIR = _REPO_ROOT / "phase3"
if str(_PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE3_DIR))

import report as _rpt  # noqa: E402  (phase3/report.py)
import score as _sc  # noqa: E402  (phase3/score.py)

from . import s_report

# Re-export the reused pieces so callers have one import site.
RecordScore = _sc.RecordScore
load_ground_truth = _sc.load_ground_truth
summarize = _sc.summarize
false_positive_summary = _sc.false_positive_summary
wilson_interval = _rpt.wilson_interval
build_cells_with_wilson = _rpt.build_cells_with_wilson
MEASURES = _rpt.MEASURES

RUN_STATUSES = ("valid", "parse_failure", "harness_error")


# --------------------------------------------------------------------------
# Loading and scoring (reuses phase3/score.py unchanged)
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def score_records(records: list[dict], ground_truth: dict[str, dict] | None = None) -> list[RecordScore]:
    if ground_truth is None:
        ground_truth = load_ground_truth()
    return _sc.score_records(records, ground_truth)


def score_jsonl(path: Path, ground_truth: dict[str, dict] | None = None) -> list[RecordScore]:
    return score_records(load_jsonl(path), ground_truth)


# --------------------------------------------------------------------------
# Rates with Wilson intervals (reuses phase3/report.py)
# --------------------------------------------------------------------------

def cells_with_wilson(scores: list[RecordScore]) -> dict:
    """Per (variant|channel|payload_lang) k/n for each measure plus
    worst_case_M1, each measure carrying a Wilson 95% interval. This is
    exactly the `cells` object stored in phase3_results.json and in each
    phase3b arm's `measures`, so re-scoring the stored records reproduces
    those cells bit for bit."""
    return build_cells_with_wilson(summarize(scores)["cells"])


def rate_with_wilson(k: int, n: int) -> dict:
    """A single n/N rate with its Wilson 95% interval, for any rate the
    report shows that is not one of the per-cell measure rates (e.g. an S
    rate, a paired-change share, a run-status share)."""
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "wilson_low": round(low, 4), "wilson_high": round(high, 4)}


# --------------------------------------------------------------------------
# Run status against planned runs (Section 3) + recovered calls (Section 4)
# --------------------------------------------------------------------------

def run_status_counts(records: Iterable[dict], planned_runs: int | None = None) -> dict:
    """Counts of each outcome_status against the planned-run total, plus
    the recovered-from-content call count (reused from s_report). `missing`
    is planned_runs - observed when a planned total is given (runs that
    never produced a record); it is None otherwise. Does not itself retry
    or re-run anything -- a harness_error record here is one the harness
    already re-ran once and that failed again (Section 2 denominators)."""
    records = list(records)
    counts = {status: 0 for status in RUN_STATUSES}
    for r in records:
        status = r.get("outcome_status")
        if status in counts:
            counts[status] += 1
        else:
            counts.setdefault("other", 0)
            counts["other"] += 1
    observed = len(records)
    return {
        "planned": planned_runs,
        "observed": observed,
        "missing": None if planned_runs is None else planned_runs - observed,
        "by_status": counts,
        "recovered_call_count": s_report.recovered_call_count(records),
    }


# --------------------------------------------------------------------------
# Paired comparison (Section 3): direction of change between two conditions
# --------------------------------------------------------------------------

def _default_condition(score: RecordScore) -> str:
    return score.channel


def _default_pair_key(score: RecordScore) -> tuple:
    # Everything that identifies the same receipt+variant across the two
    # conditions being compared, with the condition (channel) left out.
    return (score.receipt_set, score.variant, score.payload_id, score.payload_lang, score.receipt_id)


def paired_change(
    scores: list[RecordScore],
    measure: str,
    cond_a: str,
    cond_b: str,
    *,
    condition: Callable[[RecordScore], object] = _default_condition,
    pair_key: Callable[[RecordScore], tuple] = _default_pair_key,
) -> dict:
    """Per-receipt direction of change of one measure from `cond_a` to
    `cond_b` (e.g. image vs text on the same receipt).

    For every pair_key seen with BOTH conditions and a non-None measure on
    each side, the direction is:
      - "gained"    a=0 -> b=1
      - "lost"      a=1 -> b=0
      - "unchanged" a == b
    A pair is dropped (counted in `dropped`, with the reason) when a
    pair_key lacks one of the two conditions, has the same condition twice
    (a duplicate), or either side's measure is None (excluded /
    harness_error). The per-receipt directions are returned so the report
    can show the change on each receipt, not only the totals."""
    if measure not in MEASURES and measure != "worst_case_M1":
        raise ValueError(f"unknown measure {measure!r}")

    buckets: dict[tuple, dict[object, list[RecordScore]]] = {}
    for s in scores:
        buckets.setdefault(pair_key(s), {}).setdefault(condition(s), []).append(s)

    directions: dict[str, int] = {"gained": 0, "lost": 0, "unchanged": 0}
    per_receipt: list[dict] = []
    dropped: list[dict] = []

    for key, by_cond in buckets.items():
        a_list, b_list = by_cond.get(cond_a), by_cond.get(cond_b)
        if not a_list or not b_list:
            dropped.append({"pair_key": key, "reason": "missing_condition"})
            continue
        if len(a_list) > 1 or len(b_list) > 1:
            dropped.append({"pair_key": key, "reason": "duplicate_condition"})
            continue
        a_val, b_val = getattr(a_list[0], measure), getattr(b_list[0], measure)
        if a_val is None or b_val is None:
            dropped.append({"pair_key": key, "reason": "measure_none"})
            continue
        a_val, b_val = bool(a_val), bool(b_val)
        if a_val == b_val:
            direction = "unchanged"
        elif not a_val and b_val:
            direction = "gained"
        else:
            direction = "lost"
        directions[direction] += 1
        per_receipt.append({"pair_key": key, "a": int(a_val), "b": int(b_val), "direction": direction})

    return {
        "measure": measure,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_pairs": len(per_receipt),
        "directions": directions,
        "per_receipt": per_receipt,
        "dropped": dropped,
        "n_dropped": len(dropped),
    }
