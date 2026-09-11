from datetime import datetime, timezone

import pytest

from mpe.analysis.bootstrap import (
    BootstrapComparabilityError,
    bootstrap_paired_difference,
    bootstrap_run_rate,
)
from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _record(
    run_id: str, item_id: str, parallel_item_id: str, refusal: bool | None, label: str = "harmful", **overrides
) -> ResultRecord:
    fields = dict(
        run_id=run_id,
        experiment_name="exp",
        lineage="instruct",
        stage=CheckpointStage.SFT,
        hf_repo_id="x",
        revision="main",
        language="en",
        benchmark="polyguard_prompts",
        item_id=item_id,
        parallel_item_id=parallel_item_id,
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="p",
        completion="c",
        is_parseable=True,
        language_match=True,
        mc_correct=None,
        refusal_label=None,
        behavior_matches_expected=None,
        expected_label=label,
        scorer_response_refusal=refusal,
        scorer_parse_ok=True if refusal is not None else None,
        generation_config=GenerationConfig(),
        created_at=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return ResultRecord(**fields)


class TestBootstrapRunRate:
    def test_point_estimate_matches_the_observed_rate(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        # 4 harmful items: 3 refused -> rate 0.75
        store.write([_record("run1", f"i{i}", str(i), i < 3) for i in range(4)])

        result = bootstrap_run_rate(store, "run1", condition="harmful", n_resamples=500, seed=1)
        assert result.point_estimate == 0.75
        assert result.n_items == 4

    def test_ci_bounds_are_ordered_and_contain_plausible_range(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        pattern = [True, True, False, True, False, False, True, False, True, True]
        store.write([_record("run1", f"i{i}", str(i), pattern[i]) for i in range(10)])

        result = bootstrap_run_rate(store, "run1", condition="harmful", n_resamples=1000, seed=1)
        assert 0.0 <= result.ci_low <= result.point_estimate <= result.ci_high <= 1.0

    def test_same_seed_is_fully_reproducible(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        pattern = [True, False, True, True, False, False, True, False]
        store.write([_record("run1", f"i{i}", str(i), pattern[i]) for i in range(8)])

        a = bootstrap_run_rate(store, "run1", condition="harmful", n_resamples=500, seed=42)
        b = bootstrap_run_rate(store, "run1", condition="harmful", n_resamples=500, seed=42)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_different_seeds_can_give_different_cis(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        pattern = [True, False, True, True, False, False, True, False]
        store.write([_record("run1", f"i{i}", str(i), pattern[i]) for i in range(8)])

        a = bootstrap_run_rate(store, "run1", condition="harmful", n_resamples=500, seed=1)
        b = bootstrap_run_rate(store, "run1", condition="harmful", n_resamples=500, seed=2)
        # Not asserting they always differ (could coincide), just that both
        # are valid, independently-seeded, deterministic computations.
        assert a.seed == 1 and b.seed == 2

    def test_missing_run_id_raises(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            bootstrap_run_rate(store, "does-not-exist", condition="harmful")

    def test_unharmful_condition_uses_unharmful_subset_only(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record("run1", "h1", "1", True, label="harmful"),
                _record("run1", "h2", "2", True, label="harmful"),
                _record("run1", "u1", "3", False, label="unharmful"),
                _record("run1", "u2", "4", True, label="unharmful"),
            ]
        )
        result = bootstrap_run_rate(store, "run1", condition="unharmful", n_resamples=200, seed=1)
        assert result.n_items == 2
        assert result.point_estimate == 0.5


class TestBootstrapPairedDifference:
    def test_point_estimate_is_the_observed_delta(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        # run_a (en) 2/2 refused = 1.0, run_b (ko) 0/2 refused = 0.0 -> delta
        # -1.0. Same stage, different language -- a valid family A pair.
        store.write([_record("run_a", f"a{i}", str(i), True, language="en") for i in range(2)])
        store.write([_record("run_b", f"b{i}", str(i), False, language="ko") for i in range(2)])

        result = bootstrap_paired_difference(store, "run_a", "run_b", condition="harmful", n_resamples=200, seed=1)
        assert result.point_estimate == -1.0
        assert result.n_paired_items == 2

    def test_paired_resampling_respects_pairing_when_rates_are_perfectly_correlated(self, tmp_path):
        """If run_a and run_b's refusal values are IDENTICAL per
        parallel_item_id, a correctly PAIRED bootstrap must yield delta=0
        for every single resample (the same id is drawn for both sides at
        once), so the CI collapses to an exact point mass at 0. Real
        variance exists in the underlying rate (not constant True/False),
        so this isn't trivially true for an unpaired/independent
        resampling -- it specifically pins that pairing is preserved."""
        store = ResultStore(results_dir=tmp_path)
        pattern = [True, True, False, True, False, False, True, False, True, True]
        store.write([_record("run_a", f"a{i}", str(i), pattern[i], language="en") for i in range(10)])
        store.write([_record("run_b", f"b{i}", str(i), pattern[i], language="ko") for i in range(10)])

        result = bootstrap_paired_difference(store, "run_a", "run_b", condition="harmful", n_resamples=500, seed=7)
        assert result.point_estimate == 0.0
        assert result.ci_low == 0.0
        assert result.ci_high == 0.0

    def test_same_seed_is_fully_reproducible(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        pattern_a = [True, True, False, True, False, False, True, False]
        pattern_b = [False, True, False, False, False, True, True, False]
        store.write([_record("run_a", f"a{i}", str(i), pattern_a[i], language="en") for i in range(8)])
        store.write([_record("run_b", f"b{i}", str(i), pattern_b[i], language="ko") for i in range(8)])

        r1 = bootstrap_paired_difference(store, "run_a", "run_b", condition="harmful", n_resamples=500, seed=3)
        r2 = bootstrap_paired_difference(store, "run_a", "run_b", condition="harmful", n_resamples=500, seed=3)
        assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)

    def test_mismatched_parallel_ids_raise_comparability_error(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run_a", "a1", "1", True), _record("run_a", "a2", "2", True)])
        store.write([_record("run_b", "b1", "2", True), _record("run_b", "b2", "3", True)])

        with pytest.raises(BootstrapComparabilityError):
            bootstrap_paired_difference(store, "run_a", "run_b", condition="harmful", n_resamples=100, seed=1)

    def test_duplicate_parallel_ids_raise_comparability_error(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record("run_a", "a1", "1", True),
                _record("run_a", "a2", "1", True),  # duplicate parallel_item_id
                _record("run_a", "a3", "2", True),
            ]
        )
        store.write([_record("run_b", "b1", "1", True), _record("run_b", "b2", "2", True)])

        with pytest.raises(BootstrapComparabilityError):
            bootstrap_paired_difference(store, "run_a", "run_b", condition="harmful", n_resamples=100, seed=1)

    def test_cross_language_paired_comparison_is_allowed_when_aligned(self, tmp_path):
        """EN vs KO must NOT be rejected by the comparability gate -- only
        alignment.aligned matters here, not comparable_trajectory (which
        would incorrectly block every EN/KO comparison since it's never
        same_language)."""
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run_en", f"a{i}", str(i), True, language="en") for i in range(3)])
        store.write([_record("run_ko", f"b{i}", str(i), False, language="ko") for i in range(3)])

        result = bootstrap_paired_difference(store, "run_en", "run_ko", condition="harmful", n_resamples=100, seed=1)
        assert result.point_estimate == -1.0

    def test_no_matching_condition_items_raises_comparability_error(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run_a", "a1", "1", True, label="harmful", language="en")])
        store.write([_record("run_b", "b1", "1", False, label="harmful", language="ko")])

        with pytest.raises(BootstrapComparabilityError, match="No 'unharmful'"):
            bootstrap_paired_difference(store, "run_a", "run_b", condition="unharmful", n_resamples=100, seed=1)

    def test_missing_run_id_raises(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run_a", "a1", "1", True)])
        with pytest.raises(ValueError, match="not found"):
            bootstrap_paired_difference(store, "run_a", "does-not-exist", condition="harmful")


class TestPairedComparabilityFamilies:
    """Only two comparison families are allowed: same-stage cross-language
    (A) or same-language cross-stage within COMPARABLE_TRAJECTORY_STAGES
    (B). Everything else -- mixed language+stage differences, different
    lineage/benchmark, Base involvement -- must be rejected."""

    @staticmethod
    def _write_run(store, run_id, ids, *, language="en", stage=CheckpointStage.SFT, lineage="instruct", benchmark="polyguard_prompts"):
        store.write(
            [
                _record(
                    run_id,
                    f"{run_id}_{i}",
                    pid,
                    True,
                    language=language,
                    stage=stage,
                    lineage=lineage,
                    benchmark=benchmark,
                )
                for i, pid in enumerate(ids)
            ]
        )

    def test_sft_en_vs_sft_ko_allowed(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "sft_en", ids, language="en", stage=CheckpointStage.SFT)
        self._write_run(store, "sft_ko", ids, language="ko", stage=CheckpointStage.SFT)
        bootstrap_paired_difference(store, "sft_en", "sft_ko", condition="harmful", n_resamples=50, seed=1)

    def test_dpo_en_vs_dpo_ko_allowed(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "dpo_en", ids, language="en", stage=CheckpointStage.DPO)
        self._write_run(store, "dpo_ko", ids, language="ko", stage=CheckpointStage.DPO)
        bootstrap_paired_difference(store, "dpo_en", "dpo_ko", condition="harmful", n_resamples=50, seed=1)

    def test_sft_en_vs_dpo_en_allowed(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "sft_en", ids, language="en", stage=CheckpointStage.SFT)
        self._write_run(store, "dpo_en", ids, language="en", stage=CheckpointStage.DPO)
        bootstrap_paired_difference(store, "sft_en", "dpo_en", condition="harmful", n_resamples=50, seed=1)

    def test_dpo_ko_vs_rlvr_ko_allowed(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "dpo_ko", ids, language="ko", stage=CheckpointStage.DPO)
        self._write_run(store, "rlvr_ko", ids, language="ko", stage=CheckpointStage.RLVR)
        bootstrap_paired_difference(store, "dpo_ko", "rlvr_ko", condition="harmful", n_resamples=50, seed=1)

    def test_sft_en_vs_rlvr_ko_rejected_mixed_language_and_stage(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "sft_en", ids, language="en", stage=CheckpointStage.SFT)
        self._write_run(store, "rlvr_ko", ids, language="ko", stage=CheckpointStage.RLVR)
        with pytest.raises(BootstrapComparabilityError, match="BOTH"):
            bootstrap_paired_difference(store, "sft_en", "rlvr_ko", condition="harmful", n_resamples=50, seed=1)

    def test_different_benchmark_rejected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "sft_en", ids, language="en", stage=CheckpointStage.SFT, benchmark="polyguard_prompts")
        self._write_run(store, "sft_ko", ids, language="ko", stage=CheckpointStage.SFT, benchmark="xstest")
        with pytest.raises(BootstrapComparabilityError, match="benchmark"):
            bootstrap_paired_difference(store, "sft_en", "sft_ko", condition="harmful", n_resamples=50, seed=1)

    def test_different_lineage_rejected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "sft_en", ids, language="en", stage=CheckpointStage.SFT, lineage="instruct")
        self._write_run(store, "sft_ko", ids, language="ko", stage=CheckpointStage.SFT, lineage="think")
        with pytest.raises(BootstrapComparabilityError, match="lineage"):
            bootstrap_paired_difference(store, "sft_en", "sft_ko", condition="harmful", n_resamples=50, seed=1)

    def test_base_vs_sft_rejected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        self._write_run(store, "base_en", ids, language="en", stage=CheckpointStage.BASE)
        self._write_run(store, "sft_en", ids, language="ko", stage=CheckpointStage.SFT)
        with pytest.raises(BootstrapComparabilityError, match="COMPARABLE_TRAJECTORY_STAGES"):
            bootstrap_paired_difference(store, "base_en", "sft_en", condition="harmful", n_resamples=50, seed=1)

    def test_existing_misalignment_rejection_still_works(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        self._write_run(store, "sft_en", ["1", "2"], language="en", stage=CheckpointStage.SFT)
        self._write_run(store, "sft_ko", ["2", "3"], language="ko", stage=CheckpointStage.SFT)
        with pytest.raises(BootstrapComparabilityError, match="not aligned"):
            bootstrap_paired_difference(store, "sft_en", "sft_ko", condition="harmful", n_resamples=50, seed=1)

    def test_existing_duplicate_rejection_still_works(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record("sft_en", "e1", "1", True, language="en", stage=CheckpointStage.SFT),
                _record("sft_en", "e2", "1", True, language="en", stage=CheckpointStage.SFT),
            ]
        )
        self._write_run(store, "sft_ko", ["1"], language="ko", stage=CheckpointStage.SFT)
        with pytest.raises(BootstrapComparabilityError, match="not aligned"):
            bootstrap_paired_difference(store, "sft_en", "sft_ko", condition="harmful", n_resamples=50, seed=1)
