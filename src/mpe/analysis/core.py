"""Deterministic run-inspection utilities: summarize and compare stored
ResultRecords. Everything here is a pure composition of existing pieces
(ResultStore, ResultRecord's own fields, mpe.metrics.core's aggregate_rate
and expected_refusal_for_label, mpe.checkpoints.schema's trajectory-
comparability constants) -- no new scoring, parsing, or validation logic
is introduced.

Listing runs deliberately has no wrapper here: ResultStore.list_runs()
already does exactly that, and a same-signature passthrough would add a
name to remember without adding behavior.
"""

from __future__ import annotations

from mpe.analysis.schema import RunComparison, RunMetricDeltas, RunSummary, SampleAlignment
from mpe.checkpoints.schema import COMPARABLE_TRAJECTORY_STAGES, STAGE_ORDER
from mpe.metrics.core import aggregate_rate, expected_refusal_for_label
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _read_or_raise(store: ResultStore, run_id: str) -> list[ResultRecord]:
    records = store.read(run_id)
    if not records:
        raise ValueError(f"Run '{run_id}' not found (no records at {store.path_for_run(run_id)})")
    return records


def _summarize(run_id: str, records: list[ResultRecord]) -> RunSummary:
    harmful_records = [r for r in records if expected_refusal_for_label(r.expected_label) is True]
    unharmful_records = [r for r in records if expected_refusal_for_label(r.expected_label) is False]

    return RunSummary(
        run_id=run_id,
        total_records=len(records),
        parseable_rate=aggregate_rate([r.is_parseable for r in records]),
        language_match_rate=aggregate_rate([r.language_match for r in records]),
        mc_accuracy=aggregate_rate([r.mc_correct for r in records]),
        behavior_match_rate=aggregate_rate([r.behavior_matches_expected for r in records]),
        scorer_parse_rate=aggregate_rate([r.scorer_parse_ok for r in records]),
        scorer_refusal_rate=aggregate_rate([r.scorer_response_refusal for r in records]),
        scorer_harmful_response_rate=aggregate_rate([r.scorer_response_harmful for r in records]),
        harmful_prompt_count=len(harmful_records),
        unharmful_prompt_count=len(unharmful_records),
        harmful_prompt_refusal_rate=aggregate_rate([r.scorer_response_refusal for r in harmful_records]),
        unharmful_prompt_refusal_rate=aggregate_rate([r.scorer_response_refusal for r in unharmful_records]),
    )


def get_run_summary(store: ResultStore, run_id: str) -> RunSummary:
    """Deterministic aggregate metrics for one run. Raises ValueError if
    the run has no stored records -- ResultStore.write() never persists an
    empty batch, so an existing run always has >=1 record; an empty read()
    unambiguously means the run_id doesn't exist."""
    records = _read_or_raise(store, run_id)
    return _summarize(run_id, records)


def _has_duplicate_parallel_ids(records: list[ResultRecord]) -> bool:
    """True if any (language, parallel_item_id) pair repeats within one
    run. Deliberately NOT keyed on parallel_item_id alone: a single run
    covering multiple languages legitimately has one record per language
    for the same parallel_item_id, and that must not be flagged as a
    duplicate."""
    keys = [(r.language, r.parallel_item_id) for r in records]
    return len(keys) != len(set(keys))


def _check_alignment(records_a: list[ResultRecord], records_b: list[ResultRecord]) -> SampleAlignment:
    ids_a = {r.parallel_item_id for r in records_a}
    ids_b = {r.parallel_item_id for r in records_b}
    has_dup_a = _has_duplicate_parallel_ids(records_a)
    has_dup_b = _has_duplicate_parallel_ids(records_b)
    return SampleAlignment(
        run_a_count=len(records_a),
        run_b_count=len(records_b),
        has_duplicate_parallel_ids_in_run_a=has_dup_a,
        has_duplicate_parallel_ids_in_run_b=has_dup_b,
        # Converting to sets to compare IDs would silently hide duplicates
        # AND would let a multi-language run look aligned with a
        # single-language run of the same bare IDs (e.g. EN+KO for {1,2}
        # is 4 records; EN-only for {1,2} is 2 records, but both have
        # parallel_item_id set {"1","2"}) -- so duplicates and record
        # counts are both checked separately, and everything must hold
        # for aligned=True.
        aligned=(
            ids_a == ids_b
            and not has_dup_a
            and not has_dup_b
            and len(records_a) == len(records_b)
        ),
        missing_from_run_a=sorted(ids_b - ids_a),
        missing_from_run_b=sorted(ids_a - ids_b),
    )


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _build_trajectory_warning(
    aligned: bool,
    same_lineage: bool,
    same_language: bool,
    same_benchmark: bool,
    stages_comparable: bool,
    stages: list,
) -> str | None:
    reasons = []
    if not aligned:
        reasons.append("sample is not exactly aligned (mismatched parallel_item_ids and/or duplicates)")
    if not same_lineage:
        reasons.append("runs cover different lineages")
    if not same_language:
        reasons.append("runs cover different languages (cross-language comparison)")
    if not same_benchmark:
        reasons.append("runs cover different benchmarks")
    if not stages_comparable:
        reasons.append(
            "stage(s) outside COMPARABLE_TRAJECTORY_STAGES are involved "
            f"(stages compared: {[s.value for s in stages]}); Base uses a "
            "different generation protocol (raw continuation, no chat "
            "template) from SFT/DPO/RLVR's shared chat template"
        )
    if not reasons:
        return None
    return "Not a clean trajectory comparison: " + "; ".join(reasons) + "."


def compare_runs(store: ResultStore, run_a: str, run_b: str) -> RunComparison:
    """Compare two runs: sample alignment and trajectory-comparability
    checks first, deterministic metric deltas only where alignment holds.

    - alignment (via parallel_item_id, with duplicates checked separately)
      gates `deltas`: comparing rates computed over different or
      duplicated underlying items would not be meaningful.
    - `comparable_trajectory` is a stricter, additional flag requiring
      same lineage/language/benchmark and only COMPARABLE_TRAJECTORY_STAGES
      stages, on top of alignment. It does NOT gate `deltas` -- a
      cross-language or Base-involving comparison still returns its real
      numbers if the sample aligns, but can never be mistaken for a clean
      SFT->DPO->RLVR trajectory delta.
    """
    records_a = _read_or_raise(store, run_a)
    records_b = _read_or_raise(store, run_b)

    alignment = _check_alignment(records_a, records_b)

    stages = sorted({r.stage for r in records_a} | {r.stage for r in records_b}, key=STAGE_ORDER.index)
    same_lineage = {r.lineage for r in records_a} == {r.lineage for r in records_b}
    languages_a = {r.language for r in records_a}
    languages_b = {r.language for r in records_b}
    # A run spanning multiple languages can never be "same_language" as
    # another run, even if the language *sets* happen to be identical --
    # a same-language trajectory comparison means one consistent language
    # throughout both runs, not matching sets of languages.
    same_language = len(languages_a) == 1 and languages_a == languages_b
    same_benchmark = {r.benchmark for r in records_a} == {r.benchmark for r in records_b}
    stages_comparable = all(stage in COMPARABLE_TRAJECTORY_STAGES for stage in stages)

    comparable_trajectory = (
        alignment.aligned and same_lineage and same_language and same_benchmark and stages_comparable
    )
    trajectory_warning = _build_trajectory_warning(
        alignment.aligned, same_lineage, same_language, same_benchmark, stages_comparable, stages
    )

    summary_a = _summarize(run_a, records_a)
    summary_b = _summarize(run_b, records_b)

    deltas = None
    if alignment.aligned:
        deltas = RunMetricDeltas(
            parseable_rate=_delta(summary_a.parseable_rate, summary_b.parseable_rate),
            language_match_rate=_delta(summary_a.language_match_rate, summary_b.language_match_rate),
            mc_accuracy=_delta(summary_a.mc_accuracy, summary_b.mc_accuracy),
            behavior_match_rate=_delta(summary_a.behavior_match_rate, summary_b.behavior_match_rate),
            scorer_parse_rate=_delta(summary_a.scorer_parse_rate, summary_b.scorer_parse_rate),
            scorer_refusal_rate=_delta(summary_a.scorer_refusal_rate, summary_b.scorer_refusal_rate),
            scorer_harmful_response_rate=_delta(
                summary_a.scorer_harmful_response_rate, summary_b.scorer_harmful_response_rate
            ),
            harmful_prompt_refusal_rate=_delta(
                summary_a.harmful_prompt_refusal_rate, summary_b.harmful_prompt_refusal_rate
            ),
            unharmful_prompt_refusal_rate=_delta(
                summary_a.unharmful_prompt_refusal_rate, summary_b.unharmful_prompt_refusal_rate
            ),
        )

    return RunComparison(
        run_a=run_a,
        run_b=run_b,
        alignment=alignment,
        stages_compared=stages,
        same_lineage=same_lineage,
        same_language=same_language,
        same_benchmark=same_benchmark,
        stages_comparable=stages_comparable,
        comparable_trajectory=comparable_trajectory,
        trajectory_warning=trajectory_warning,
        summary_a=summary_a,
        summary_b=summary_b,
        deltas=deltas,
    )
