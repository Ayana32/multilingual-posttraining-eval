import csv
from datetime import datetime, timezone

from mpe.analysis.audit import extract_audit_candidates, write_audit_candidates_csv
from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _record(run_id: str, item_id: str, parallel_item_id: str, **overrides) -> ResultRecord:
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
        finish_reason="stop",
        is_parseable=True,
        language_match=True,
        mc_correct=None,
        refusal_label="refusal",
        behavior_matches_expected=True,
        expected_label="harmful",
        scorer_name="polyguard-qwen-smol",
        scorer_prompt_harmful=True,
        scorer_response_refusal=True,
        scorer_response_harmful=False,
        scorer_parse_ok=True,
        generation_config=GenerationConfig(),
        created_at=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return ResultRecord(**fields)


class TestExtractAuditCandidates:
    def test_clean_record_is_not_selected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "1")])  # every field innocuous
        assert extract_audit_candidates(store) == []

    def test_scorer_response_harmful_is_selected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "1", scorer_response_harmful=True)])
        candidates = extract_audit_candidates(store)
        assert len(candidates) == 1
        assert "scorer_response_harmful" in candidates[0].reasons

    def test_prompt_harm_label_disagreement_is_selected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        # expected_label="harmful" (-> True) but scorer says prompt not harmful
        store.write([_record("run1", "i1", "1", expected_label="harmful", scorer_prompt_harmful=False)])
        candidates = extract_audit_candidates(store)
        assert len(candidates) == 1
        assert "prompt_harm_label_disagreement" in candidates[0].reasons

    def test_no_disagreement_flagged_when_labels_agree(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "1", expected_label="harmful", scorer_prompt_harmful=True)])
        candidates = extract_audit_candidates(store)
        assert candidates == []

    def test_language_mismatch_is_selected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "1", language_match=False)])
        candidates = extract_audit_candidates(store)
        assert "language_mismatch" in candidates[0].reasons

    def test_truncated_completion_is_selected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "1", finish_reason="length")])
        candidates = extract_audit_candidates(store)
        assert "truncated" in candidates[0].reasons

    def test_known_suspicious_id_is_selected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "476")])
        candidates = extract_audit_candidates(store)
        assert "known_suspicious_id" in candidates[0].reasons

    def test_en_ko_scorer_disagreement_flags_both_records(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [_record("run_en", "i1", "7", language="en", scorer_response_refusal=True, stage=CheckpointStage.SFT)]
        )
        store.write(
            [_record("run_ko", "i2", "7", language="ko", scorer_response_refusal=False, stage=CheckpointStage.SFT)]
        )
        candidates = extract_audit_candidates(store, run_ids=["run_en", "run_ko"])
        assert len(candidates) == 2
        assert all("en_ko_scorer_disagreement" in c.reasons for c in candidates)
        assert {c.language for c in candidates} == {"en", "ko"}

    def test_en_ko_agreement_is_not_flagged(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [_record("run_en", "i1", "7", language="en", scorer_response_refusal=True, stage=CheckpointStage.SFT)]
        )
        store.write(
            [_record("run_ko", "i2", "7", language="ko", scorer_response_refusal=True, stage=CheckpointStage.SFT)]
        )
        candidates = extract_audit_candidates(store, run_ids=["run_en", "run_ko"])
        assert candidates == []

    def test_refusal_flip_across_stages_flags_all_involved_records(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [_record("run_sft", "i1", "8", stage=CheckpointStage.SFT, scorer_response_refusal=True)]
        )
        store.write(
            [_record("run_dpo", "i2", "8", stage=CheckpointStage.DPO, scorer_response_refusal=False)]
        )
        store.write(
            [_record("run_rlvr", "i3", "8", stage=CheckpointStage.RLVR, scorer_response_refusal=False)]
        )
        candidates = extract_audit_candidates(store, run_ids=["run_sft", "run_dpo", "run_rlvr"])
        assert len(candidates) == 3
        assert all("refusal_flip_across_stages" in c.reasons for c in candidates)

    def test_no_flip_when_refusal_is_stable_across_stages(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run_sft", "i1", "8", stage=CheckpointStage.SFT, scorer_response_refusal=True)])
        store.write([_record("run_dpo", "i2", "8", stage=CheckpointStage.DPO, scorer_response_refusal=True)])
        candidates = extract_audit_candidates(store, run_ids=["run_sft", "run_dpo"])
        assert candidates == []

    def test_base_stage_is_excluded_from_stage_flip_detection(self, tmp_path):
        """Base uses a different generation protocol -- a Base-vs-SFT
        refusal difference must never be read as a genuine cross-stage
        flip (mpe.checkpoints.schema.COMPARABLE_TRAJECTORY_STAGES)."""
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run_base", "i1", "8", stage=CheckpointStage.BASE, scorer_response_refusal=True)])
        store.write([_record("run_sft", "i2", "8", stage=CheckpointStage.SFT, scorer_response_refusal=False)])
        candidates = extract_audit_candidates(store, run_ids=["run_base", "run_sft"])
        assert candidates == []

    def test_a_record_can_accumulate_multiple_reasons(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [_record("run1", "i1", "1", scorer_response_harmful=True, finish_reason="length", language_match=False)]
        )
        candidates = extract_audit_candidates(store)
        assert len(candidates) == 1
        assert set(candidates[0].reasons) == {"scorer_response_harmful", "truncated", "language_mismatch"}

    def test_run_ids_filter_restricts_which_runs_are_scanned(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "1", scorer_response_harmful=True)])
        store.write([_record("run2", "i1", "1", scorer_response_harmful=True)])
        candidates = extract_audit_candidates(store, run_ids=["run1"])
        assert len(candidates) == 1
        assert candidates[0].run_id == "run1"

    def test_results_are_sorted_deterministically(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run_z", "iz", "9", scorer_response_harmful=True)])
        store.write([_record("run_a", "ia", "1", scorer_response_harmful=True)])
        candidates = extract_audit_candidates(store, run_ids=["run_z", "run_a"])
        assert [c.run_id for c in candidates] == ["run_a", "run_z"]

    def test_multiple_en_runs_for_the_same_item_do_not_silently_overwrite_each_other(self, tmp_path):
        """Two different run_ids both covering (lineage, stage, benchmark,
        parallel_item_id="7") in English, plus one Korean record, must NOT
        silently pick one of the two EN records for the disagreement
        check -- both EN records and the KO record must instead be flagged
        ambiguous, and no en_ko_scorer_disagreement may fire."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [_record("run_en_1", "i1", "7", language="en", stage=CheckpointStage.SFT, scorer_response_refusal=True)]
        )
        store.write(
            [_record("run_en_2", "i2", "7", language="en", stage=CheckpointStage.SFT, scorer_response_refusal=False)]
        )
        store.write(
            [_record("run_ko", "i3", "7", language="ko", stage=CheckpointStage.SFT, scorer_response_refusal=True)]
        )
        candidates = extract_audit_candidates(store, run_ids=["run_en_1", "run_en_2", "run_ko"])

        assert len(candidates) == 3
        assert all("ambiguous_cross_run_group" in c.reasons for c in candidates)
        assert all("en_ko_scorer_disagreement" not in c.reasons for c in candidates)

    def test_separate_experiment_cohorts_do_not_create_a_false_en_ko_disagreement(self, tmp_path):
        """Two runs from clearly different experiment cohorts (different
        experiment_name, e.g. a Phase 2A pilot run and an unrelated
        re-run) both happen to cover the same English item at the same
        stage; one Korean run also covers it. Without the ambiguity guard,
        an arbitrary one of the two English records would be picked and
        could produce a fabricated disagreement against the Korean
        record -- this must not happen."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(
                    "pilot_run",
                    "i1",
                    "7",
                    language="en",
                    stage=CheckpointStage.SFT,
                    experiment_name="pilot_sft_en_polyguard",
                    scorer_response_refusal=True,
                )
            ]
        )
        store.write(
            [
                _record(
                    "phase2b_run",
                    "i2",
                    "7",
                    language="en",
                    stage=CheckpointStage.SFT,
                    experiment_name="phase2b_sft_en_polyguard",
                    scorer_response_refusal=False,
                )
            ]
        )
        store.write(
            [_record("ko_run", "i3", "7", language="ko", stage=CheckpointStage.SFT, scorer_response_refusal=False)]
        )
        candidates = extract_audit_candidates(store, run_ids=["pilot_run", "phase2b_run", "ko_run"])

        assert all("en_ko_scorer_disagreement" not in c.reasons for c in candidates)
        assert all("ambiguous_cross_run_group" in c.reasons for c in candidates)

    def test_overlapping_phase2a_phase2b_style_runs_do_not_create_false_stage_flips(self, tmp_path):
        """Two separate SFT runs (different cohorts) plus one DPO run, all
        covering the same item -- must not fabricate a stage flip from an
        arbitrarily-chosen SFT record."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(
                    "phase2a_sft",
                    "i1",
                    "8",
                    stage=CheckpointStage.SFT,
                    experiment_name="pilot_sft_en_polyguard",
                    scorer_response_refusal=True,
                )
            ]
        )
        store.write(
            [
                _record(
                    "phase2b_sft",
                    "i2",
                    "8",
                    stage=CheckpointStage.SFT,
                    experiment_name="phase2b_sft_en_polyguard",
                    scorer_response_refusal=False,
                )
            ]
        )
        store.write([_record("phase2b_dpo", "i3", "8", stage=CheckpointStage.DPO, scorer_response_refusal=False)])
        candidates = extract_audit_candidates(store, run_ids=["phase2a_sft", "phase2b_sft", "phase2b_dpo"])

        assert all("refusal_flip_across_stages" not in c.reasons for c in candidates)
        assert all("ambiguous_cross_run_group" in c.reasons for c in candidates)

    def test_single_record_reasons_still_fire_alongside_ambiguity(self, tmp_path):
        """A record that is both part of an ambiguous cross-run group AND
        individually flaggable (e.g. truncated) must carry both reasons --
        single-record reasons are computed independently of the cross-run
        ambiguity pass."""
        store = ResultStore(results_dir=tmp_path)
        store.write(
            [
                _record(
                    "run_en_1", "i1", "7", language="en", stage=CheckpointStage.SFT,
                    scorer_response_refusal=True, finish_reason="length",
                )
            ]
        )
        store.write(
            [_record("run_en_2", "i2", "7", language="en", stage=CheckpointStage.SFT, scorer_response_refusal=False)]
        )
        store.write(
            [_record("run_ko", "i3", "7", language="ko", stage=CheckpointStage.SFT, scorer_response_refusal=True)]
        )
        candidates = extract_audit_candidates(store, run_ids=["run_en_1", "run_en_2", "run_ko"])

        truncated_candidate = next(c for c in candidates if c.run_id == "run_en_1")
        assert set(truncated_candidate.reasons) == {"ambiguous_cross_run_group", "truncated"}


class TestWriteAuditCandidatesCsv:
    def test_writes_expected_rows_and_columns(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("run1", "i1", "1", scorer_response_harmful=True, finish_reason="length")])
        candidates = extract_audit_candidates(store)

        out_path = tmp_path / "audit.csv"
        write_audit_candidates_csv(candidates, out_path)

        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["parallel_item_id"] == "1"
        assert row["run_id"] == "run1"
        assert set(row["reasons"].split(";")) == {"scorer_response_harmful", "truncated"}
        assert row["completion"] == "c"

    def test_empty_candidate_list_writes_header_only(self, tmp_path):
        out_path = tmp_path / "audit.csv"
        write_audit_candidates_csv([], out_path)
        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows == []
