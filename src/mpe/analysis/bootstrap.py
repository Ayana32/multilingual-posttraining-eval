"""Deterministic percentile-bootstrap utilities over stored ResultRecords.

No model inference, no p-values -- percentile confidence intervals only,
computed by resampling already-stored scorer verdicts. Reuses
mpe.metrics.core.aggregate_rate for every rate computation (the point
estimate itself comes from mpe.analysis.core.get_run_summary) and
mpe.analysis.core.compare_runs for the paired-comparison alignment check --
no metric or alignment logic is duplicated here, only the resampling loop
and percentile computation are new.
"""

from __future__ import annotations

import random
from typing import Literal

from mpe.analysis.core import compare_runs, get_run_summary
from mpe.analysis.schema import BootstrapEstimate, PairedBootstrapDifference, RunComparison
from mpe.metrics.core import aggregate_rate, expected_refusal_for_label
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore

Condition = Literal["harmful", "unharmful"]

_CONDITION_VALUE = {"harmful": True, "unharmful": False}


class BootstrapComparabilityError(ValueError):
    """Raised instead of silently computing an invalid confidence interval
    when a paired bootstrap's prerequisites (exact parallel_item_id
    alignment between the two runs) are not met."""


def _condition_matches(expected_label: str | None, condition: Condition) -> bool:
    return expected_refusal_for_label(expected_label) is _CONDITION_VALUE[condition]


def _validate_paired_comparability(
    comparison: RunComparison,
    records_a: list[ResultRecord],
    records_b: list[ResultRecord],
    run_a: str,
    run_b: str,
) -> None:
    """Allows exactly the two intended Phase 2B paired-comparison families:

    A. Cross-language: same lineage, same benchmark, same single stage,
       different single languages (e.g. SFT-EN vs SFT-KO).
    B. Cross-stage: same lineage, same benchmark, same single language,
       both stages within COMPARABLE_TRAJECTORY_STAGES (stages may
       differ, e.g. SFT-EN vs DPO-EN).

    Rejects: mixing both a language difference and a stage difference in
    the same comparison (e.g. SFT-EN vs RLVR-KO), different lineage,
    different benchmark, Base involvement, and (already handled by
    compare_runs()) misaligned or duplicate-containing samples.

    Reuses compare_runs()'s already-computed same_lineage/same_benchmark/
    stages_comparable/alignment -- only the language/stage *family* check
    (which of the two allowed shapes this is) is new logic.
    """
    if not comparison.alignment.aligned:
        raise BootstrapComparabilityError(
            f"Cannot compute a paired bootstrap between '{run_a}' and '{run_b}': "
            f"samples are not aligned ({comparison.alignment.model_dump()})."
        )
    if not comparison.same_lineage:
        raise BootstrapComparabilityError(
            f"Cannot compute a paired bootstrap between '{run_a}' and '{run_b}': different lineages."
        )
    if not comparison.same_benchmark:
        raise BootstrapComparabilityError(
            f"Cannot compute a paired bootstrap between '{run_a}' and '{run_b}': different benchmarks."
        )
    if not comparison.stages_comparable:
        raise BootstrapComparabilityError(
            f"Cannot compute a paired bootstrap between '{run_a}' and '{run_b}': stage(s) outside "
            f"COMPARABLE_TRAJECTORY_STAGES are involved (stages compared: "
            f"{[s.value for s in comparison.stages_compared]}) -- Base uses a different generation "
            "protocol and must never be treated as part of the SFT->DPO->RLVR trajectory."
        )

    stages_a = {r.stage for r in records_a}
    stages_b = {r.stage for r in records_b}
    languages_a = {r.language for r in records_a}
    languages_b = {r.language for r in records_b}

    if len(stages_a) != 1 or len(stages_b) != 1:
        raise BootstrapComparabilityError(
            f"Cannot compute a paired bootstrap between '{run_a}' and '{run_b}': each run must cover "
            f"a single, consistent stage (run_a stages: {stages_a}, run_b stages: {stages_b})."
        )
    if len(languages_a) != 1 or len(languages_b) != 1:
        raise BootstrapComparabilityError(
            f"Cannot compute a paired bootstrap between '{run_a}' and '{run_b}': each run must cover "
            f"a single, consistent language (run_a languages: {languages_a}, run_b languages: {languages_b})."
        )

    same_stage = stages_a == stages_b
    same_language = languages_a == languages_b

    if same_stage and same_language:
        raise BootstrapComparabilityError(
            f"'{run_a}' and '{run_b}' are the same stage and language -- nothing to compare. A paired "
            "bootstrap needs either a language difference (same stage) or a stage difference (same "
            "language), not neither."
        )
    if not same_stage and not same_language:
        raise BootstrapComparabilityError(
            f"Cannot compute a paired bootstrap between '{run_a}' and '{run_b}': they differ in BOTH "
            f"language ({languages_a} vs {languages_b}) and stage ({stages_a} vs {stages_b}). Only a "
            "same-stage cross-language comparison or a same-language cross-stage comparison is valid "
            "-- not both differences mixed together in one paired comparison."
        )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy's default 'linear'
    method) -- implemented in pure Python so this utility introduces no new
    dependency beyond what the project already has."""
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty distribution")
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _bootstrap_ci_single(
    values: list[bool | None], n_resamples: int, seed: int
) -> tuple[float | None, float | None]:
    if aggregate_rate(values) is None:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    rates: list[float] = []
    for _ in range(n_resamples):
        resampled = rng.choices(values, k=n)
        rate = aggregate_rate(resampled)
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None, None
    rates.sort()
    return _percentile(rates, 2.5), _percentile(rates, 97.5)


def bootstrap_run_rate(
    store: ResultStore,
    run_id: str,
    *,
    condition: Condition,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapEstimate:
    """Per-run rate + 95% percentile bootstrap CI for one condition
    (harmful-prompt or unharmful-prompt refusal rate).

    The point estimate is get_run_summary()'s already-computed rate (real
    data, no resampling); the CI is derived by resampling that condition's
    per-item scorer_response_refusal values with replacement.
    """
    summary = get_run_summary(store, run_id)  # validates run_id exists
    records = store.read(run_id)
    condition_records = [r for r in records if _condition_matches(r.expected_label, condition)]
    point_estimate = (
        summary.harmful_prompt_refusal_rate
        if condition == "harmful"
        else summary.unharmful_prompt_refusal_rate
    )
    values = [r.scorer_response_refusal for r in condition_records]
    ci_low, ci_high = _bootstrap_ci_single(values, n_resamples, seed)
    return BootstrapEstimate(
        metric=f"{condition}_prompt_refusal_rate",
        run_id=run_id,
        n_items=len(condition_records),
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=n_resamples,
        seed=seed,
    )


def bootstrap_paired_difference(
    store: ResultStore,
    run_a: str,
    run_b: str,
    *,
    condition: Condition,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapDifference:
    """Paired-by-parallel_item_id difference (run_b's rate minus run_a's)
    + 95% percentile bootstrap CI, for one condition.

    Allows exactly the two intended Phase 2B comparison families (see
    _validate_paired_comparability): a same-stage cross-language comparison
    (e.g. SFT-EN vs SFT-KO) or a same-language cross-stage comparison
    within COMPARABLE_TRAJECTORY_STAGES (e.g. SFT-EN vs DPO-EN). A
    comparison that mixes both a language difference and a stage
    difference (e.g. SFT-EN vs RLVR-KO) is rejected, as is a mismatched
    lineage/benchmark, Base involvement, or a misaligned/duplicate sample.

    Reuses compare_runs() for alignment/lineage/benchmark/stage-
    comparability -- no alignment logic is duplicated here, only the
    language/stage *family* check (which of the two allowed comparison
    shapes this is) is new.
    """
    comparison = compare_runs(store, run_a, run_b)  # validates both run_ids exist
    records_a = list(store.read(run_a))
    records_b = list(store.read(run_b))
    _validate_paired_comparability(comparison, records_a, records_b, run_a, run_b)

    records_a = {r.parallel_item_id: r for r in records_a}
    records_b = {r.parallel_item_id: r for r in records_b}
    # alignment.aligned already guarantees records_a.keys() == records_b.keys()
    # with no duplicates, so this is a clean 1:1 pairing by construction.
    shared_ids = sorted(records_a)
    condition_ids = [i for i in shared_ids if _condition_matches(records_a[i].expected_label, condition)]

    if not condition_ids:
        raise BootstrapComparabilityError(
            f"No '{condition}' items in the aligned sample between '{run_a}' and '{run_b}'."
        )

    refusal_a = {i: records_a[i].scorer_response_refusal for i in condition_ids}
    refusal_b = {i: records_b[i].scorer_response_refusal for i in condition_ids}

    observed_rate_a = aggregate_rate([refusal_a[i] for i in condition_ids])
    observed_rate_b = aggregate_rate([refusal_b[i] for i in condition_ids])
    point_estimate = (
        None if observed_rate_a is None or observed_rate_b is None else observed_rate_b - observed_rate_a
    )

    rng = random.Random(seed)
    n = len(condition_ids)
    deltas: list[float] = []
    for _ in range(n_resamples):
        resampled = rng.choices(condition_ids, k=n)
        rate_a = aggregate_rate([refusal_a[i] for i in resampled])
        rate_b = aggregate_rate([refusal_b[i] for i in resampled])
        if rate_a is not None and rate_b is not None:
            deltas.append(rate_b - rate_a)

    if not deltas:
        ci_low, ci_high = None, None
    else:
        deltas.sort()
        ci_low, ci_high = _percentile(deltas, 2.5), _percentile(deltas, 97.5)

    return PairedBootstrapDifference(
        metric=f"{condition}_prompt_refusal_rate",
        run_a=run_a,
        run_b=run_b,
        n_paired_items=len(condition_ids),
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=n_resamples,
        seed=seed,
    )
