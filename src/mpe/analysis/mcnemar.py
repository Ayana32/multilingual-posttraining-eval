"""Exact (binomial) McNemar test over two paired runs' stored scorer verdicts.

Kept separate from mpe.analysis.bootstrap, which deliberately reports CIs
only. McNemar answers a different, narrower question: given the same
items scored in two conditions, is the imbalance between discordant pairs
(refused in A but not B vs. refused in B but not A) larger than chance?
It uses only the discordant pairs, which is what makes it the right test
for a paired design where most items behave the same in both conditions.

The exact binomial form is used (not the chi-square approximation) because
pilot-scale discordant counts are small. Pure Python, no new dependency.

Comparability rules are NOT reimplemented: this reuses compare_runs() and
the bootstrap module's _validate_paired_comparability(), so exactly the same
two comparison families are allowed (same-stage cross-language, or
same-language cross-stage within COMPARABLE_TRAJECTORY_STAGES).
"""

from __future__ import annotations

from math import comb
from typing import Literal

from pydantic import BaseModel

from mpe.analysis.bootstrap import _condition_matches, _validate_paired_comparability
from mpe.analysis.core import compare_runs
from mpe.storage.store import ResultStore

Condition = Literal["harmful", "unharmful"]


class McNemarResult(BaseModel):
    """Paired 2x2 table of refusal verdicts for one condition, plus the
    exact two-sided McNemar p-value computed from its discordant cells."""

    metric: str
    run_a: str
    run_b: str
    n_paired_items: int
    """Condition-matching pairs where BOTH verdicts are non-None."""
    n_excluded_pairs: int
    """Condition-matching pairs dropped because either verdict was None
    (scorer not run or parse failure). Reported, never silently hidden."""
    both_refused: int
    only_a_refused: int
    only_b_refused: int
    neither_refused: int
    p_value_exact: float
    """Two-sided exact binomial p-value on the discordant pairs. 1.0 when
    there are no discordant pairs (no evidence of any difference)."""


def exact_mcnemar_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value: under H0 each discordant pair is
    equally likely to fall either way, so the smaller discordant count is
    Binomial(n, 0.5). Doubling the one-sided tail, capped at 1."""
    if only_a < 0 or only_b < 0:
        raise ValueError("discordant counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def mcnemar_paired(
    store: ResultStore,
    run_a: str,
    run_b: str,
    *,
    condition: Condition = "harmful",
) -> McNemarResult:
    comparison = compare_runs(store, run_a, run_b)  # validates both run_ids exist
    records_a_list = list(store.read(run_a))
    records_b_list = list(store.read(run_b))
    _validate_paired_comparability(comparison, records_a_list, records_b_list, run_a, run_b)

    records_a = {r.parallel_item_id: r for r in records_a_list}
    records_b = {r.parallel_item_id: r for r in records_b_list}
    condition_ids = [
        i for i in sorted(records_a) if _condition_matches(records_a[i].expected_label, condition)
    ]

    both = only_a = only_b = neither = excluded = 0
    for i in condition_ids:
        a = records_a[i].scorer_response_refusal
        b = records_b[i].scorer_response_refusal
        if a is None or b is None:
            excluded += 1
        elif a and b:
            both += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1

    return McNemarResult(
        metric=f"{condition}_prompt_refusal_rate",
        run_a=run_a,
        run_b=run_b,
        n_paired_items=both + only_a + only_b + neither,
        n_excluded_pairs=excluded,
        both_refused=both,
        only_a_refused=only_a,
        only_b_refused=only_b,
        neither_refused=neither,
        p_value_exact=exact_mcnemar_p(only_a, only_b),
    )
