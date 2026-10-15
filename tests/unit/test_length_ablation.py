"""Focused deterministic tests for mpe.analysis.length_ablation, synthetic
records only. See docs/phase2a-length-ablation.md for the plan this
module implements."""

from datetime import datetime, timezone

import pytest

from mpe.analysis.length_ablation import (
    KNOWN_INVALID_RUN_IDS,
    LengthAblationPair,
    check_g0,
    check_g1a,
    check_g2,
    check_transitions,
    compute_length_interaction,
    extract_review_candidates,
)
from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _record(
    run_id: str,
    item_id: str,
    parallel_item_id: str,
    refusal: bool | None,
    *,
    language: str = "en",
    label: str = "harmful",
    completion: str = "full response text",
    finish_reason: str = "stop",
    harmful_response: bool = False,
    max_new_tokens: int = 200,
    **overrides,
) -> ResultRecord:
    fields = dict(
        run_id=run_id,
        experiment_name="exp",
        lineage="instruct",
        stage=CheckpointStage.SFT,
        hf_repo_id="allenai/Olmo-3-7B-Instruct-SFT",
        revision="main",
        language=language,
        benchmark="polyguard_prompts",
        item_id=item_id,
        parallel_item_id=parallel_item_id,
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="p",
        completion=completion,
        finish_reason=finish_reason,
        is_parseable=True,
        language_match=True,
        mc_correct=None,
        refusal_label=None,
        behavior_matches_expected=None,
        expected_label=label,
        scorer_name="polyguard-qwen-smol",
        scorer_response_refusal=refusal,
        scorer_response_harmful=harmful_response,
        scorer_parse_ok=True if refusal is not None else None,
        generation_config=GenerationConfig(max_new_tokens=max_new_tokens, seed=42),
        created_at=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return ResultRecord(**fields)


def _write_pair(store, run_200, run_1024, ids, refusals_200, refusals_1024, **kw):
    store.write(
        [_record(run_200, f"{run_200}_{i}", pid, r, max_new_tokens=200, **kw) for i, (pid, r) in enumerate(zip(ids, refusals_200))]
    )
    store.write(
        [_record(run_1024, f"{run_1024}_{i}", pid, r, max_new_tokens=1024, **kw) for i, (pid, r) in enumerate(zip(ids, refusals_1024))]
    )


class TestG0:
    def test_passes_for_a_valid_same_everything_except_length_pair(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = ["1", "2", "3"]
        _write_pair(store, "r200", "r1024", ids, [True, True, False], [True, False, False])

        result = check_g0(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.passed is True
        assert result.reason is None

    def test_detects_a_mismatched_item_set(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", True, max_new_tokens=200), _record("r200", "a2", "2", True, max_new_tokens=200)])
        store.write([_record("r1024", "b1", "2", True, max_new_tokens=1024), _record("r1024", "b2", "3", True, max_new_tokens=1024)])

        result = check_g0(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.passed is False
        assert "not aligned" in result.reason

    def test_detects_a_differing_non_length_generation_parameter(self, tmp_path):
        """temperature differs between the two runs -- must be rejected
        even though max_new_tokens is the 'expected' difference."""
        store = ResultStore(results_dir=tmp_path)
        gen_a = GenerationConfig(max_new_tokens=200, temperature=0.0, seed=42)
        gen_b = GenerationConfig(max_new_tokens=1024, temperature=0.7, seed=42)
        store.write([_record("r200", "a1", "1", True, generation_config=gen_a)])
        store.write([_record("r1024", "b1", "1", True, generation_config=gen_b)])

        result = check_g0(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.passed is False
        assert "temperature" in result.reason or "differ" in result.reason

    def test_detects_a_differing_checkpoint(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", True, hf_repo_id="allenai/Olmo-3-7B-Instruct-SFT")])
        store.write([_record("r1024", "b1", "1", True, hf_repo_id="allenai/Olmo-3-7B-Instruct-DPO")])

        result = check_g0(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.passed is False
        assert "checkpoint" in result.reason

    def test_known_invalid_run_id_is_rejected_as_run_200(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        invalid_id = next(iter(KNOWN_INVALID_RUN_IDS))
        store.write([_record(invalid_id, "a1", "1", None, completion="")])
        store.write([_record("r1024", "b1", "1", True, max_new_tokens=1024)])

        result = check_g0(
            store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200=invalid_id, run_1024="r1024")
        )
        assert result.passed is False
        assert "known-invalid" in result.reason

    def test_known_invalid_run_id_is_rejected_as_run_1024(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        invalid_id = next(iter(KNOWN_INVALID_RUN_IDS))
        store.write([_record("r200", "a1", "1", True)])
        store.write([_record(invalid_id, "b1", "1", None, completion="", max_new_tokens=1024)])

        result = check_g0(
            store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024=invalid_id)
        )
        assert result.passed is False
        assert "known-invalid" in result.reason


class TestG1a:
    def test_counts_and_rate_among_harmful_prompt_refusals_only(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        # EN: 2 harmful refusals, 1 truncated. KO: 2 harmful refusals, both truncated.
        store.write(
            [
                _record("en", "e1", "1", True, label="harmful", finish_reason="length"),
                _record("en", "e2", "2", True, label="harmful", finish_reason="stop"),
                _record("en", "e3", "3", False, label="unharmful", finish_reason="length"),  # not a refusal -> excluded
            ]
        )
        store.write(
            [
                _record("ko", "k1", "1", True, label="harmful", finish_reason="length", language="ko"),
                _record("ko", "k2", "2", True, label="harmful", finish_reason="length", language="ko"),
            ]
        )
        result = check_g1a(store, CheckpointStage.SFT, 1024, "en", "ko", gated=True)
        assert (result.en.truncated, result.en.total) == (1, 2)
        assert (result.ko.truncated, result.ko.total) == (2, 2)
        assert result.abs_diff_pp == pytest.approx(50.0)

    def test_boundary_case_exactly_at_max_truncated_passes(self, tmp_path):
        """Revised 2026-10-15: the pass rule is an absolute truncated-count
        cap (default 1 per language), not a rate threshold. EN and KO each
        have exactly 1 truncated harmful refusal -> at the cap -> passes."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record("en", f"e{i}", str(i), True, label="harmful", finish_reason="length" if i == 0 else "stop")
                for i in range(10)
            ]
        )
        store.write(
            [
                _record("ko", f"k{i}", str(i), True, label="harmful", language="ko",
                         finish_reason="length" if i == 0 else "stop")
                for i in range(10)
            ]
        )
        result = check_g1a(store, CheckpointStage.SFT, 1024, "en", "ko", gated=True)
        assert (result.en.truncated, result.ko.truncated) == (1, 1)
        assert result.passed is True

    def test_boundary_case_just_over_max_truncated_fails(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("en", f"e{i}", str(i), True, label="harmful", finish_reason="stop") for i in range(10)])
        ko_records = [
            _record("ko", f"k{i}", str(i), True, label="harmful", language="ko", finish_reason="length")
            for i in range(2)
        ] + [
            _record("ko", f"k{i}", str(i), True, label="harmful", language="ko", finish_reason="stop")
            for i in range(2, 10)
        ]
        store.write(ko_records)
        result = check_g1a(store, CheckpointStage.SFT, 1024, "en", "ko", gated=True)
        assert result.ko.truncated == 2
        assert result.passed is False

    def test_passes_despite_large_abs_diff_pp_when_counts_are_within_cap(self, tmp_path):
        """abs_diff_pp is descriptive only since the 2026-10-15 revision:
        small KO denominators (7-9, as in the real pilot data) can swing
        the rate by double digits from a single truncated item. A large
        abs_diff_pp must not fail the gate when both truncated counts are
        within the cap -- mirrors the real SFT 0/23 EN vs ~1/9 KO shape."""
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("en", f"e{i}", str(i), True, label="harmful", finish_reason="stop") for i in range(23)])
        store.write(
            [_record("ko", "k0", "x0", True, label="harmful", language="ko", finish_reason="length")]
            + [
                _record("ko", f"k{i}", f"x{i}", True, label="harmful", language="ko", finish_reason="stop")
                for i in range(1, 9)
            ]
        )
        result = check_g1a(store, CheckpointStage.SFT, 1024, "en", "ko", gated=True)
        assert (result.en.truncated, result.ko.truncated) == (0, 1)
        assert result.abs_diff_pp > 10.0  # the old rate threshold would have failed this
        assert result.passed is True

    def test_baseline_200_token_result_is_not_gated(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("en", "e1", "1", True, label="harmful", finish_reason="length")])
        store.write([_record("ko", "k1", "1", True, label="harmful", language="ko", finish_reason="length")])
        result = check_g1a(store, CheckpointStage.SFT, 200, "en", "ko", gated=False)
        assert result.gated is False
        assert result.passed is None

    def test_zero_denominator_on_both_sides_passes_vacuously_with_flag_set(self, tmp_path):
        """Revised: zero refusals means zero truncated refusals, which
        trivially satisfies the count cap. passed must be a concrete
        True (not None), with zero_refusal_denominator=True flagging that
        the pass is vacuous -- otherwise all(passed) over gate results
        would read a missing denominator as a silent failure."""
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("en", "e1", "1", False, label="unharmful")])  # no harmful refusals at all
        store.write([_record("ko", "k1", "1", False, label="unharmful", language="ko")])
        result = check_g1a(store, CheckpointStage.SFT, 1024, "en", "ko", gated=True)
        assert result.en.rate is None
        assert result.abs_diff_pp is None
        assert result.zero_refusal_denominator is True
        assert result.passed is True

    def test_zero_denominator_on_one_side_still_evaluates_the_other(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("en", "e1", "1", False, label="unharmful")])  # EN: no harmful refusals
        ko_records = [
            _record("ko", f"k{i}", str(i), True, label="harmful", language="ko", finish_reason="length")
            for i in range(2)
        ]
        store.write(ko_records)
        result = check_g1a(store, CheckpointStage.SFT, 1024, "en", "ko", gated=True)
        assert result.en.total == 0
        assert result.zero_refusal_denominator is True
        assert result.ko.truncated == 2
        assert result.passed is False  # KO side genuinely exceeds the cap


class TestG2PrefixFidelity:
    def test_full_agreement_when_1024_starts_with_200_text(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", True, completion="The answer is no.")])
        store.write([_record("r1024", "b1", "1", True, completion="The answer is no. Here is more detail.", max_new_tokens=1024)])

        result = check_g2(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.n_agree == 1
        assert result.agreement_rate == 1.0
        assert result.divergences == []

    def test_one_divergent_item_is_reported_with_its_index(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record("r200", "a1", "1", True, completion="Hello world"),
                _record("r200", "a2", "2", True, completion="Same prefix here"),
            ]
        )
        store.write(
            [
                _record("r1024", "b1", "1", True, completion="Hello WORLD different", max_new_tokens=1024),  # diverges at index 6
                _record("r1024", "b2", "2", True, completion="Same prefix here and more", max_new_tokens=1024),  # agrees
            ]
        )

        result = check_g2(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.n_items == 2
        assert result.n_agree == 1
        assert result.agreement_rate == 0.5
        assert len(result.divergences) == 1
        assert result.divergences[0].parallel_item_id == "1"
        assert result.divergences[0].first_divergence_index == 6

    def test_1024_shorter_than_200_counts_as_divergence(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", True, completion="A full long response")])
        store.write([_record("r1024", "b1", "1", True, completion="A full", max_new_tokens=1024)])

        result = check_g2(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.n_agree == 0
        assert result.divergences[0].first_divergence_index == len("A full")
        assert result.n_real_divergences == 1
        assert result.n_boundary_only_divergences == 0

    def test_trailing_replacement_char_is_stripped_and_true_agreement_detected(self, tmp_path):
        """A multi-byte Korean syllable split at the 200-token cut decodes
        its orphaned bytes as a trailing U+FFFD even though the underlying
        tokens agree with the 1024-token run. Stripping it before
        comparison must recover full agreement, not a false divergence."""
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", True, completion="안녕하세요�", language="ko")])
        store.write(
            [_record("r1024", "b1", "1", True, completion="안녕하세요. 등당합니다.", language="ko", max_new_tokens=1024)]
        )

        result = check_g2(store, LengthAblationPair(stage=CheckpointStage.SFT, language="ko", run_200="r200", run_1024="r1024"))
        assert result.n_agree == 1
        assert result.divergences == []
        assert result.n_boundary_only_divergences == 0
        assert result.n_real_divergences == 0

    def test_divergence_confined_to_final_character_is_boundary_only(self, tmp_path):
        """Orphaned bytes at the cut can decode as a *different* valid
        character instead of U+FFFD. If the only mismatch is within the
        final character of the (U+FFFD-stripped) 200-token text, it must
        be reported as boundary-only, not a real content divergence."""
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", True, completion="Hello world X")])
        store.write([_record("r1024", "b1", "1", True, completion="Hello world Y and more", max_new_tokens=1024)])

        result = check_g2(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.n_agree == 0
        assert len(result.divergences) == 1
        assert result.divergences[0].is_boundary_only is True
        assert result.n_boundary_only_divergences == 1
        assert result.n_real_divergences == 0

    def test_divergence_before_final_character_is_a_real_divergence(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", True, completion="Hello world")])
        store.write([_record("r1024", "b1", "1", True, completion="Hello WORLD different", max_new_tokens=1024)])

        result = check_g2(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert len(result.divergences) == 1
        assert result.divergences[0].is_boundary_only is False
        assert result.n_real_divergences == 1
        assert result.n_boundary_only_divergences == 0


class TestTransitionStratification:
    def test_transitions_are_split_by_truncation_at_200(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                # truncated at 200: refusal -> non-refusal
                _record("r200", "a1", "1", True, finish_reason="length"),
                # not truncated at 200: refusal -> refusal
                _record("r200", "a2", "2", True, finish_reason="stop"),
                # not truncated at 200: non-refusal -> refusal
                _record("r200", "a3", "3", False, finish_reason="stop"),
            ]
        )
        store.write(
            [
                _record("r1024", "b1", "1", False, max_new_tokens=1024),
                _record("r1024", "b2", "2", True, max_new_tokens=1024),
                _record("r1024", "b3", "3", True, max_new_tokens=1024),
            ]
        )

        result = check_transitions(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.truncated_at_200.refusal_to_nonrefusal == 1
        assert result.truncated_at_200.refusal_to_refusal == 0
        assert result.not_truncated_at_200.refusal_to_refusal == 1
        assert result.not_truncated_at_200.nonrefusal_to_refusal == 1

    def test_none_verdicts_are_excluded_not_miscounted(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", None, finish_reason="stop")])
        store.write([_record("r1024", "b1", "1", True, max_new_tokens=1024)])

        result = check_transitions(store, LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024"))
        assert result.not_truncated_at_200.excluded_none_verdict == 1
        total = (
            result.not_truncated_at_200.refusal_to_refusal
            + result.not_truncated_at_200.refusal_to_nonrefusal
            + result.not_truncated_at_200.nonrefusal_to_refusal
            + result.not_truncated_at_200.nonrefusal_to_nonrefusal
        )
        assert total == 0


class TestLengthInteraction:
    def test_d_sign_reflects_a_shrinking_gap(self, tmp_path):
        """EN refuses less at 1024 while KO refuses more at 1024 -> the
        EN-KO gap shrinks -> D should be negative."""
        store = ResultStore(results_dir=tmp_path)
        ids = [str(i) for i in range(10)]
        # 200: EN all refuse, KO none refuse -> EN-KO = 1.0
        store.write([_record("en200", f"e{i}", ids[i], True, language="en") for i in range(10)])
        store.write([_record("ko200", f"k{i}", ids[i], False, language="ko") for i in range(10)])
        # 1024: EN half refuse, KO half refuse -> EN-KO = 0.0
        store.write([_record("en1024", f"e{i}", ids[i], i < 5, language="en", max_new_tokens=1024) for i in range(10)])
        store.write([_record("ko1024", f"k{i}", ids[i], i < 5, language="ko", max_new_tokens=1024) for i in range(10)])

        pair_en = LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="en200", run_1024="en1024")
        pair_ko = LengthAblationPair(stage=CheckpointStage.SFT, language="ko", run_200="ko200", run_1024="ko1024")
        result = compute_length_interaction(store, pair_en, pair_ko, n_resamples=500, seed=1)

        assert result.point_estimate == pytest.approx(-1.0)
        assert result.ci_low is not None and result.ci_high is not None

    def test_bootstrap_ci_is_deterministic_for_a_fixed_seed(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        ids = [str(i) for i in range(10)]
        pattern_en200 = [True, False, True, True, False, False, True, False, True, True]
        pattern_ko200 = [False, True, False, False, False, True, True, False, False, True]
        pattern_en1024 = [True, True, False, True, False, False, True, False, True, False]
        pattern_ko1024 = [False, False, False, True, False, True, True, True, False, True]
        store.write([_record("en200", f"e{i}", ids[i], pattern_en200[i], language="en") for i in range(10)])
        store.write([_record("ko200", f"k{i}", ids[i], pattern_ko200[i], language="ko") for i in range(10)])
        store.write([_record("en1024", f"e{i}", ids[i], pattern_en1024[i], language="en", max_new_tokens=1024) for i in range(10)])
        store.write([_record("ko1024", f"k{i}", ids[i], pattern_ko1024[i], language="ko", max_new_tokens=1024) for i in range(10)])

        pair_en = LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="en200", run_1024="en1024")
        pair_ko = LengthAblationPair(stage=CheckpointStage.SFT, language="ko", run_200="ko200", run_1024="ko1024")

        r1 = compute_length_interaction(store, pair_en, pair_ko, n_resamples=500, seed=7)
        r2 = compute_length_interaction(store, pair_en, pair_ko, n_resamples=500, seed=7)
        assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)

    def test_misaligned_four_runs_raise_comparability_error(self, tmp_path):
        from mpe.analysis.bootstrap import BootstrapComparabilityError

        store = ResultStore(results_dir=tmp_path)
        store.write([_record("en200", "e1", "1", True, label="harmful", language="en")])
        store.write([_record("ko200", "k1", "1", True, label="harmful", language="ko")])
        store.write([_record("en1024", "e1", "1", True, label="harmful", language="en", max_new_tokens=1024)])
        store.write([_record("ko1024", "k1", "2", True, label="harmful", language="ko", max_new_tokens=1024)])  # different id!

        pair_en = LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="en200", run_1024="en1024")
        pair_ko = LengthAblationPair(stage=CheckpointStage.SFT, language="ko", run_200="ko200", run_1024="ko1024")
        with pytest.raises(BootstrapComparabilityError):
            compute_length_interaction(store, pair_en, pair_ko, n_resamples=50, seed=1)


class TestReviewCandidateScope:
    def test_truncated_nonrefusal_1024_only_flagged_on_harmful_prompt_items(self, tmp_path):
        """Revised 2026-10-15: truncated_nonrefusal_1024 must not fire on
        an unharmful-prompt item -- G1b only asks whether harmful content
        preceded truncation, which isn't a question for unharmful prompts."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record("r200", "a1", "1", False, label="harmful", finish_reason="stop"),
                _record("r200", "a2", "2", False, label="unharmful", finish_reason="stop"),
            ]
        )
        store.write(
            [
                _record("r1024", "b1", "1", False, label="harmful", finish_reason="length", max_new_tokens=1024),
                _record("r1024", "b2", "2", False, label="unharmful", finish_reason="length", max_new_tokens=1024),
            ]
        )

        pair = LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024")
        candidates = extract_review_candidates(store, [pair])

        flagged = {c.parallel_item_id: c.reasons for c in candidates}
        assert "truncated_nonrefusal_1024" in flagged.get("1", [])
        assert "1" in flagged
        assert "2" not in flagged  # unharmful-prompt item: not flagged at all under this rule

    def test_annotator_csv_omits_reasons_and_key_csv_carries_it(self, tmp_path):
        from mpe.analysis.length_ablation import write_review_candidates_csv, write_review_key_csv

        store = ResultStore(results_dir=tmp_path)
        store.write([_record("r200", "a1", "1", False, label="harmful", finish_reason="stop")])
        store.write(
            [_record("r1024", "b1", "1", False, label="harmful", finish_reason="length", max_new_tokens=1024)]
        )
        pair = LengthAblationPair(stage=CheckpointStage.SFT, language="en", run_200="r200", run_1024="r1024")
        candidates = extract_review_candidates(store, [pair])
        assert candidates and "truncated_nonrefusal_1024" in candidates[0].reasons

        annotator_csv = write_review_candidates_csv(candidates, tmp_path / "annotator.csv")
        key_csv = write_review_key_csv(candidates, tmp_path / "key.csv")

        annotator_text = annotator_csv.read_text(encoding="utf-8-sig")
        assert "reasons" not in annotator_text.splitlines()[0]
        assert "truncated_nonrefusal_1024" not in annotator_text

        key_text = key_csv.read_text(encoding="utf-8-sig")
        assert "reasons" in key_text.splitlines()[0]
        assert "truncated_nonrefusal_1024" in key_text
