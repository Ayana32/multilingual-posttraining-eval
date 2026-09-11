from __future__ import annotations

from pydantic import BaseModel

from mpe.checkpoints.schema import CheckpointStage


class RunSummary(BaseModel):
    """Deterministic aggregate metrics for one run's stored ResultRecords.

    Every rate is a mean of a boolean field across the run's records via
    mpe.metrics.core.aggregate_rate (None-excluding, None if nothing
    applicable was observed) -- no new statistic is invented here, this is
    purely a composition of that existing primitive over ResultRecord's
    already-stored fields.

    harmful_prompt_refusal_rate / unharmful_prompt_refusal_rate reuse
    expected_label (via mpe.metrics.core.expected_refusal_for_label, the
    same harm/safe vocabulary used everywhere else in this project) and
    scorer_response_refusal -- packaging the exact "harmful-prompt refusal
    rate" / "unharmful-prompt refusal rate" split that every pilot
    comparison in this project has computed by hand.
    """

    run_id: str
    total_records: int
    parseable_rate: float | None
    language_match_rate: float | None
    mc_accuracy: float | None
    behavior_match_rate: float | None
    scorer_parse_rate: float | None
    scorer_refusal_rate: float | None
    scorer_harmful_response_rate: float | None
    harmful_prompt_count: int
    unharmful_prompt_count: int
    harmful_prompt_refusal_rate: float | None
    unharmful_prompt_refusal_rate: float | None


class SampleAlignment(BaseModel):
    """Whether two runs cover the exact same parallel_item_id set, with no
    duplicates in either run and matching record counts.

    This is the precondition for any meaningful metric delta between two
    runs -- e.g. an EN pilot and a KO pilot, or an SFT run and a DPO run,
    are only comparable if they scored the same underlying items exactly
    once each. Record counts matter in addition to the ID set: a
    multi-language run can share a single-language run's bare
    parallel_item_id set while covering more actual records.
    """

    run_a_count: int
    run_b_count: int
    has_duplicate_parallel_ids_in_run_a: bool
    """True if any (language, parallel_item_id) pair repeats within run_a
    -- the same item scored twice for the same language, a genuine
    anomaly. Not flagged for a legitimate multi-language run, which has
    one record per language for the same parallel_item_id by design."""
    has_duplicate_parallel_ids_in_run_b: bool
    aligned: bool
    """True only if run_a and run_b cover the exact same parallel_item_id
    set, neither run has duplicates, AND both runs have the same record
    count. The count check matters on top of the ID-set check: a
    multi-language run (e.g. EN+KO for parallel IDs {1,2} = 4 records) has
    the exact same bare parallel_item_id set as a single-language run
    covering the same IDs (e.g. EN-only for {1,2} = 2 records), but they
    are not the same sample. Converting to a set to compare IDs would
    silently hide both duplicates and this record-count mismatch, so both
    are checked separately."""
    missing_from_run_a: list[str]
    """parallel_item_ids present in run_b but not run_a, sorted."""
    missing_from_run_b: list[str]
    """parallel_item_ids present in run_a but not run_b, sorted."""


class RunMetricDeltas(BaseModel):
    """run_b's RunSummary rate minus run_a's, per metric.

    A field is None if either run's corresponding RunSummary field is None
    (nothing observed to subtract), never a fabricated 0.
    """

    parseable_rate: float | None
    language_match_rate: float | None
    mc_accuracy: float | None
    behavior_match_rate: float | None
    scorer_parse_rate: float | None
    scorer_refusal_rate: float | None
    scorer_harmful_response_rate: float | None
    harmful_prompt_refusal_rate: float | None
    unharmful_prompt_refusal_rate: float | None


class RunComparison(BaseModel):
    """Full result of comparing two runs: comparability checks first, then
    (only where they pass) deterministic metric deltas.

    `deltas` is gated ONLY by alignment.aligned: a cross-language or
    Base-involving comparison still gets real deltas if the underlying
    sample aligns exactly, consistent with how this project has always
    still reported Base's and cross-language numbers -- never hidden,
    always clearly labeled as not a clean trajectory effect.

    `comparable_trajectory` is the stricter, additional flag for "is this
    specifically a clean same-language SFT->DPO->RLVR trajectory
    comparison" -- it additionally requires same_lineage, same_language,
    same_benchmark, and stages_comparable on top of alignment.aligned.
    Every contributing check is exposed as its own field so the reason a
    comparison isn't trajectory-clean is explicit, never hidden behind a
    single opaque boolean.
    """

    run_a: str
    run_b: str
    alignment: SampleAlignment
    stages_compared: list[CheckpointStage]
    same_lineage: bool
    same_language: bool
    """True only if BOTH runs contain a single, matching language -- not
    merely identical language *sets*. A run spanning multiple languages
    (e.g. {en, ko}) is never same_language as anything, even another run
    spanning the identical {en, ko} set: a trajectory comparison means one
    consistent language throughout."""
    same_benchmark: bool
    stages_comparable: bool
    """False if any stage among stages_compared is outside
    COMPARABLE_TRAJECTORY_STAGES (i.e. Base is involved). Base is evaluated
    via raw prompt continuation, a categorically different generation
    protocol from SFT/DPO/RLVR's shared chat template."""
    comparable_trajectory: bool
    """True only if alignment.aligned AND same_lineage AND same_language
    AND same_benchmark AND stages_comparable all hold -- the full bar for
    "clean same-language post-training trajectory comparison". False for
    any cross-language, cross-lineage, cross-benchmark, misaligned/
    duplicate-containing, or Base-involving comparison, regardless of
    whether deltas are still returned."""
    trajectory_warning: str | None
    """Explicit, human-readable listing of every failing check (not just
    the first one) when comparable_trajectory is False; None when it's
    True."""
    summary_a: RunSummary
    summary_b: RunSummary
    deltas: RunMetricDeltas | None
    """None when alignment.aligned is False."""


class BootstrapEstimate(BaseModel):
    """A single run's rate for one condition, with a percentile bootstrap
    95% CI computed by resampling the underlying per-item scorer verdicts.

    point_estimate is the actual observed rate (via RunSummary, i.e.
    mpe.metrics.core.aggregate_rate over the real data) -- the bootstrap
    resampling is used only to derive ci_low/ci_high, never to replace the
    point estimate with a resampled statistic.
    """

    metric: str
    """"harmful_prompt_refusal_rate" | "unharmful_prompt_refusal_rate"."""
    run_id: str
    n_items: int
    """Number of condition-matching records the estimate is computed over."""
    point_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    """None/None when n_items has no scorer verdicts to resample."""
    n_resamples: int
    seed: int


class PairedBootstrapDifference(BaseModel):
    """A paired-by-parallel_item_id difference (run_b's rate minus run_a's)
    for one condition, with a percentile bootstrap 95% CI.

    Every bootstrap resample draws the same set of parallel_item_ids for
    both runs simultaneously (not independently), preserving the pairing --
    this is what makes it a *paired* difference rather than two independent
    single-run bootstraps subtracted from each other.
    """

    metric: str
    run_a: str
    run_b: str
    n_paired_items: int
    """Number of condition-matching ids present in both runs' aligned sample."""
    point_estimate: float | None
    """Observed run_b rate minus observed run_a rate (not a bootstrap statistic)."""
    ci_low: float | None
    ci_high: float | None
    n_resamples: int
    seed: int


class AuditCandidate(BaseModel):
    """One ResultRecord flagged for manual scorer-reliability review, with
    every reason it was selected. A record can accumulate multiple reasons
    (e.g. both scorer_response_harmful=True and a truncated completion).

    Carries only fields needed for manual review -- see
    mpe.analysis.audit.extract_audit_candidates for the selection rules.
    """

    run_id: str
    parallel_item_id: str
    language: str
    stage: CheckpointStage
    benchmark: str
    expected_label: str | None
    scorer_prompt_harmful: bool | None
    scorer_response_refusal: bool | None
    scorer_response_harmful: bool | None
    finish_reason: str
    language_match: bool | None
    completion: str
    reasons: list[str]
    """Machine-readable reason codes, e.g. "scorer_response_harmful",
    "prompt_harm_label_disagreement", "en_ko_scorer_disagreement",
    "refusal_flip_across_stages", "language_mismatch", "truncated",
    "known_suspicious_id", "ambiguous_cross_run_group" (a cross-run
    comparison -- EN/KO or cross-stage -- was skipped because more than
    one record shared the same language/stage within the comparison
    group, e.g. two independent runs of the same cohort or overlapping
    experiment sets; see mpe.analysis.audit)."""
