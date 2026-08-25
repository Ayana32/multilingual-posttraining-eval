from datetime import datetime, timezone

import pytest

from mpe.analysis import compare_runs, get_run_summary
from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _record(run_id: str = "run1", item_id: str = "item1", parallel_item_id: str = "1", **overrides) -> ResultRecord:
    fields = dict(
        run_id=run_id,
        experiment_name="exp",
        lineage="instruct",
        stage=CheckpointStage.SFT,
        hf_repo_id="allenai/Olmo-3-7B-Instruct-SFT",
        revision="main",
        language="en",
        benchmark="polyguard_prompts",
        item_id=item_id,
        parallel_item_id=parallel_item_id,
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="prompt text",
        completion="completion text",
        is_parseable=True,
        language_match=True,
        mc_correct=None,
        refusal_label="refusal",
        behavior_matches_expected=True,
        expected_label="harmful",
        scorer_name=None,
        scorer_prompt_harmful=None,
        scorer_response_refusal=None,
        scorer_response_harmful=None,
        scorer_parse_ok=None,
        generation_config=GenerationConfig(),
        created_at=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return ResultRecord(**fields)


class TestGetRunSummary:
    def test_metric_summary_correctness(self, tmp_path):
        """Hand-computed expected rates against a small, fully-controlled
        set of records -- pins that get_run_summary is a correct
        composition of aggregate_rate over the right fields, not just
        "doesn't crash"."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(item_id="1", parallel_item_id="1", is_parseable=True, language_match=True,
                        scorer_parse_ok=True, scorer_response_refusal=True, scorer_response_harmful=False),
                _record(item_id="2", parallel_item_id="2", is_parseable=True, language_match=False,
                        scorer_parse_ok=True, scorer_response_refusal=False, scorer_response_harmful=False),
                _record(item_id="3", parallel_item_id="3", is_parseable=False, language_match=None,
                        scorer_parse_ok=False, scorer_response_refusal=None, scorer_response_harmful=None),
                _record(item_id="4", parallel_item_id="4", is_parseable=True, language_match=True,
                        scorer_parse_ok=None, scorer_response_refusal=None, scorer_response_harmful=None),
            ]
        )

        summary = get_run_summary(store, "run1")

        assert summary.run_id == "run1"
        assert summary.total_records == 4
        assert summary.parseable_rate == 0.75  # 3/4 True
        assert summary.language_match_rate == pytest.approx(2 / 3)  # True,False,None-excluded,True -> 2/3
        assert summary.mc_accuracy is None  # never set -> all None -> aggregate_rate returns None
        assert summary.scorer_parse_rate == pytest.approx(2 / 3)  # True,True,False,None-excluded
        assert summary.scorer_refusal_rate == 0.5  # True,False among the two scored items
        assert summary.scorer_harmful_response_rate == 0.0  # False,False among the two scored items

    def test_harmful_vs_unharmful_refusal_summary_correctness(self, tmp_path):
        """Reuses expected_label + scorer_response_refusal (no new scoring
        logic) to split refusal rate by harmful vs unharmful prompt --
        the exact split every pilot comparison in this project has
        computed by hand."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                # 3 harmful prompts: 2 refused, 1 not -> harmful refusal rate 2/3
                _record(item_id="h1", parallel_item_id="h1", expected_label="harmful", scorer_response_refusal=True),
                _record(item_id="h2", parallel_item_id="h2", expected_label="harmful", scorer_response_refusal=True),
                _record(item_id="h3", parallel_item_id="h3", expected_label="harmful", scorer_response_refusal=False),
                # 2 unharmful prompts: 1 refused (over-refusal), 1 not -> unharmful refusal rate 1/2
                _record(item_id="u1", parallel_item_id="u1", expected_label="unharmful", scorer_response_refusal=True),
                _record(item_id="u2", parallel_item_id="u2", expected_label="unharmful", scorer_response_refusal=False),
                # a non-refusal-vocabulary label must not be counted as either
                _record(item_id="c1", parallel_item_id="c1", expected_label="control", scorer_response_refusal=True),
            ]
        )

        summary = get_run_summary(store, "run1")

        assert summary.total_records == 6
        assert summary.harmful_prompt_count == 3
        assert summary.unharmful_prompt_count == 2
        assert summary.harmful_prompt_refusal_rate == pytest.approx(2 / 3)
        assert summary.unharmful_prompt_refusal_rate == pytest.approx(0.5)

    def test_missing_run_id_raises(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            get_run_summary(store, "does-not-exist")


class TestCompareRunsAlignment:
    def test_aligned_runs_produce_deltas(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(run_id="run_a", item_id="a1", parallel_item_id="1", scorer_response_refusal=True),
                _record(run_id="run_a", item_id="a2", parallel_item_id="2", scorer_response_refusal=True),
            ]
        )
        store.write(
            [
                _record(run_id="run_b", item_id="b1", parallel_item_id="1", scorer_response_refusal=False),
                _record(run_id="run_b", item_id="b2", parallel_item_id="2", scorer_response_refusal=True),
            ]
        )

        result = compare_runs(store, "run_a", "run_b")

        assert result.alignment.aligned is True
        assert result.alignment.run_a_count == 2
        assert result.alignment.run_b_count == 2
        assert result.alignment.missing_from_run_a == []
        assert result.alignment.missing_from_run_b == []
        assert result.alignment.has_duplicate_parallel_ids_in_run_a is False
        assert result.alignment.has_duplicate_parallel_ids_in_run_b is False
        assert result.deltas is not None
        # run_a: 2/2 refusal = 1.0, run_b: 1/2 refusal = 0.5 -> delta = -0.5
        assert result.deltas.scorer_refusal_rate == pytest.approx(-0.5)

    def test_mismatched_parallel_ids_block_deltas(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(run_id="run_a", item_id="a1", parallel_item_id="1"),
                _record(run_id="run_a", item_id="a2", parallel_item_id="2"),
            ]
        )
        store.write(
            [
                _record(run_id="run_b", item_id="b1", parallel_item_id="2"),
                _record(run_id="run_b", item_id="b2", parallel_item_id="3"),
            ]
        )

        result = compare_runs(store, "run_a", "run_b")

        assert result.alignment.aligned is False
        assert result.alignment.missing_from_run_a == ["3"]  # in run_b, not run_a
        assert result.alignment.missing_from_run_b == ["1"]  # in run_a, not run_b
        assert result.deltas is None
        assert result.comparable_trajectory is False
        # Individual summaries are still computed even when misaligned.
        assert result.summary_a.total_records == 2
        assert result.summary_b.total_records == 2

    def test_duplicate_parallel_ids_invalidate_alignment(self, tmp_path):
        """A repeated (language, parallel_item_id) pair within one run
        must not be hidden by converting IDs to a set -- it must flip
        aligned=False and block deltas, even if the ID *sets* would
        otherwise match exactly."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(run_id="run_a", item_id="a1", parallel_item_id="1"),
                _record(run_id="run_a", item_id="a2", parallel_item_id="1"),  # duplicate parallel_item_id
                _record(run_id="run_a", item_id="a3", parallel_item_id="2"),
            ]
        )
        store.write(
            [
                _record(run_id="run_b", item_id="b1", parallel_item_id="1"),
                _record(run_id="run_b", item_id="b2", parallel_item_id="2"),
            ]
        )

        result = compare_runs(store, "run_a", "run_b")

        # The bare ID sets ({"1","2"} == {"1","2"}) would look aligned --
        # the duplicate must override that.
        assert result.alignment.has_duplicate_parallel_ids_in_run_a is True
        assert result.alignment.has_duplicate_parallel_ids_in_run_b is False
        assert result.alignment.aligned is False
        assert result.deltas is None
        assert result.comparable_trajectory is False

    def test_multi_language_run_not_aligned_with_single_language_run_of_same_ids(self, tmp_path):
        """run_a: EN+KO for parallel IDs {1,2} = 4 records. run_b: EN only
        for parallel IDs {1,2} = 2 records. The bare parallel_item_id sets
        match ({"1","2"} == {"1","2"}) and neither run has an internal
        duplicate (language, parallel_item_id) pair, so the ID-set and
        duplicate checks alone would say aligned -- record-count equality
        must catch that these are not the same sample."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(run_id="run_a", item_id="a_en1", parallel_item_id="1", language="en"),
                _record(run_id="run_a", item_id="a_en2", parallel_item_id="2", language="en"),
                _record(run_id="run_a", item_id="a_ko1", parallel_item_id="1", language="ko"),
                _record(run_id="run_a", item_id="a_ko2", parallel_item_id="2", language="ko"),
            ]
        )
        store.write(
            [
                _record(run_id="run_b", item_id="b_en1", parallel_item_id="1", language="en"),
                _record(run_id="run_b", item_id="b_en2", parallel_item_id="2", language="en"),
            ]
        )

        result = compare_runs(store, "run_a", "run_b")

        assert result.alignment.run_a_count == 4
        assert result.alignment.run_b_count == 2
        assert result.alignment.has_duplicate_parallel_ids_in_run_a is False
        assert result.alignment.has_duplicate_parallel_ids_in_run_b is False
        assert result.alignment.aligned is False
        assert result.deltas is None
        assert result.comparable_trajectory is False

    def test_missing_run_id_raises(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record(run_id="run_a", parallel_item_id="1")])
        with pytest.raises(ValueError, match="not found"):
            compare_runs(store, "run_a", "does-not-exist")
        with pytest.raises(ValueError, match="not found"):
            compare_runs(store, "does-not-exist", "run_a")


class TestCompareRunsTrajectoryGuardrail:
    def test_sft_dpo_same_language_remains_trajectory_comparable(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record(run_id="run_sft", parallel_item_id="1", stage=CheckpointStage.SFT)])
        store.write([_record(run_id="run_dpo", parallel_item_id="1", stage=CheckpointStage.DPO)])

        result = compare_runs(store, "run_sft", "run_dpo")

        assert result.alignment.aligned is True
        assert result.same_lineage is True
        assert result.same_language is True
        assert result.same_benchmark is True
        assert result.stages_comparable is True
        assert result.comparable_trajectory is True
        assert result.trajectory_warning is None
        assert result.stages_compared == [CheckpointStage.SFT, CheckpointStage.DPO]
        assert result.deltas is not None

    def test_sft_en_vs_dpo_ko_not_trajectory_comparable_despite_identical_parallel_ids(self, tmp_path):
        """Same parallel_item_id (the EN/KO parallel design means this is
        expected and correct -- both runs cover "the same underlying
        prompt"), but different language must still block
        comparable_trajectory, while still allowing deltas since the
        sample itself aligns."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [_record(run_id="run_sft_en", parallel_item_id="1", language="en", stage=CheckpointStage.SFT)]
        )
        store.write(
            [_record(run_id="run_dpo_ko", parallel_item_id="1", language="ko", stage=CheckpointStage.DPO)]
        )

        result = compare_runs(store, "run_sft_en", "run_dpo_ko")

        assert result.alignment.aligned is True  # same parallel_item_id, language-agnostic
        assert result.same_language is False
        assert result.comparable_trajectory is False
        assert result.trajectory_warning is not None
        assert "language" in result.trajectory_warning
        # Cross-language, but sample aligns -> deltas still returned.
        assert result.deltas is not None

    def test_different_lineage_not_trajectory_comparable(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record(run_id="run_a", parallel_item_id="1", lineage="instruct")])
        store.write([_record(run_id="run_b", parallel_item_id="1", lineage="think")])

        result = compare_runs(store, "run_a", "run_b")

        assert result.same_lineage is False
        assert result.comparable_trajectory is False
        assert "lineage" in result.trajectory_warning

    def test_different_benchmark_not_trajectory_comparable(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record(run_id="run_a", parallel_item_id="1", benchmark="polyguard_prompts")])
        store.write([_record(run_id="run_b", parallel_item_id="1", benchmark="xstest")])

        result = compare_runs(store, "run_a", "run_b")

        assert result.same_benchmark is False
        assert result.comparable_trajectory is False
        assert "benchmark" in result.trajectory_warning

    def test_base_involvement_is_never_silently_comparable(self, tmp_path):
        """Base must not be treated as part of the causal SFT->DPO->RLVR
        trajectory: comparable_trajectory must be False and a warning must
        be present whenever Base is one of the stages compared, even
        though lineage/language/benchmark all match."""
        store = ResultStore(results_dir=tmp_path)
        store.write([_record(run_id="run_base", parallel_item_id="1", stage=CheckpointStage.BASE)])
        store.write([_record(run_id="run_sft", parallel_item_id="1", stage=CheckpointStage.SFT)])

        result = compare_runs(store, "run_base", "run_sft")

        assert result.same_lineage is True
        assert result.same_language is True
        assert result.same_benchmark is True
        assert result.stages_comparable is False
        assert result.comparable_trajectory is False
        assert result.trajectory_warning is not None
        assert "Base" in result.trajectory_warning
        assert CheckpointStage.BASE in result.stages_compared
        # Base involvement is a non-silent flag, not a hard block: aligned
        # samples still get real deltas, just clearly marked as untrusted
        # for a trajectory-effect reading.
        assert result.alignment.aligned is True
        assert result.deltas is not None

    def test_base_involvement_stages_compared_is_canonically_ordered(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record(run_id="run_rlvr", parallel_item_id="1", stage=CheckpointStage.RLVR)])
        store.write([_record(run_id="run_base", parallel_item_id="1", stage=CheckpointStage.BASE)])

        result = compare_runs(store, "run_rlvr", "run_base")

        # STAGE_ORDER = [BASE, SFT, DPO, RLVR] regardless of call order.
        assert result.stages_compared == [CheckpointStage.BASE, CheckpointStage.RLVR]


class TestSameLanguageFlag:
    @pytest.mark.parametrize(
        "languages_a, languages_b, expected_same_language",
        [
            (["en"], ["en"], True),
            (["ko"], ["ko"], True),
            (["en"], ["ko"], False),
            (["en", "ko"], ["en", "ko"], False),  # multi-language: identical SETS, not same_language
        ],
        ids=["en-vs-en", "ko-vs-ko", "en-vs-ko", "multi-vs-multi-identical-sets"],
    )
    def test_same_language_requires_both_runs_single_and_matching(
        self, tmp_path, languages_a, languages_b, expected_same_language
    ):
        """same_language must mean "both runs are single-language and that
        language matches", not "the language sets are equal" -- a run
        spanning multiple languages can never be same_language as
        anything, even another run spanning the identical language set."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(run_id="run_a", item_id=f"a_{lang}", parallel_item_id="1", language=lang)
                for lang in languages_a
            ]
        )
        store.write(
            [
                _record(run_id="run_b", item_id=f"b_{lang}", parallel_item_id="1", language=lang)
                for lang in languages_b
            ]
        )

        result = compare_runs(store, "run_a", "run_b")

        assert result.same_language is expected_same_language
