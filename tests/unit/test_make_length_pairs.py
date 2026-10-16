"""Focused deterministic tests for scripts/make_length_pairs.py, synthetic
result directories only. See docs/phase2a-length-ablation.md section 1
for the six (stage, language) conditions this script discovers pairs
for."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_length_pairs  # noqa: E402

from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _record(
    run_id: str,
    item_id: str,
    parallel_item_id: str,
    experiment_name: str,
    *,
    language: str = "en",
    max_new_tokens: int = 200,
) -> ResultRecord:
    return ResultRecord(
        run_id=run_id,
        experiment_name=experiment_name,
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
        completion="full response text",
        finish_reason="stop",
        is_parseable=True,
        language_match=True,
        mc_correct=None,
        refusal_label=None,
        behavior_matches_expected=None,
        expected_label="harmful",
        scorer_name="polyguard-qwen-smol",
        scorer_response_refusal=True,
        scorer_response_harmful=False,
        scorer_parse_ok=True,
        generation_config=GenerationConfig(max_new_tokens=max_new_tokens, seed=42),
        created_at=datetime.now(timezone.utc),
    )


def _write_200(store: ResultStore, run_id: str, *, language: str = "en") -> None:
    store.write([_record(run_id, f"{run_id}_0", "1", "irrelevant_200_experiment_name", language=language)])


def _write_1024(store: ResultStore, run_id: str, experiment_name: str, *, language: str = "en") -> None:
    store.write(
        [_record(run_id, f"{run_id}_0", "1", experiment_name, language=language, max_new_tokens=1024)]
    )


class TestBuildPairs:
    def test_builds_all_six_pairs_successfully(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        for (stage, language), run_200 in make_length_pairs.KNOWN_VALID_RUN_IDS_200.items():
            _write_200(store, run_200, language=language)
            _write_1024(
                store,
                f"pilot_{stage}_{language}_polyguard_len1024_abcdef12",
                f"pilot_{stage}_{language}_polyguard_len1024",
                language=language,
            )

        pairs = make_length_pairs.build_pairs(tmp_path)
        assert len(pairs) == 6
        by_key = {(p["stage"], p["language"]): p for p in pairs}
        for (stage, language), run_200 in make_length_pairs.KNOWN_VALID_RUN_IDS_200.items():
            pair = by_key[(stage, language)]
            assert pair["run_200"] == run_200
            assert pair["run_1024"] == f"pilot_{stage}_{language}_polyguard_len1024_abcdef12"

    def test_fails_loudly_when_1024_experiment_has_zero_matches(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        run_200 = make_length_pairs.KNOWN_VALID_RUN_IDS_200[("sft", "en")]
        _write_200(store, run_200)
        # No 1024 run written at all for sft/en.

        with pytest.raises(make_length_pairs.PairDiscoveryError, match="No run found"):
            make_length_pairs.build_pairs(tmp_path)

    def test_fails_loudly_when_1024_experiment_has_multiple_matches(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        run_200 = make_length_pairs.KNOWN_VALID_RUN_IDS_200[("sft", "en")]
        _write_200(store, run_200)
        _write_1024(store, "pilot_sft_en_polyguard_len1024_aaaaaaaa", "pilot_sft_en_polyguard_len1024")
        _write_1024(store, "pilot_sft_en_polyguard_len1024_bbbbbbbb", "pilot_sft_en_polyguard_len1024")

        with pytest.raises(make_length_pairs.PairDiscoveryError, match="found 2"):
            make_length_pairs.build_pairs(tmp_path)

    def test_fails_loudly_when_1024_run_is_known_invalid(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        run_200 = make_length_pairs.KNOWN_VALID_RUN_IDS_200[("sft", "en")]
        _write_200(store, run_200)
        invalid_id = next(iter(make_length_pairs.KNOWN_INVALID_RUN_IDS))
        _write_1024(store, invalid_id, "pilot_sft_en_polyguard_len1024")

        with pytest.raises(make_length_pairs.PairDiscoveryError, match="KNOWN_INVALID_RUN_IDS"):
            make_length_pairs.build_pairs(tmp_path)

    def test_fails_loudly_when_run_lives_under_invalid_directory(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        run_200 = make_length_pairs.KNOWN_VALID_RUN_IDS_200[("sft", "en")]
        _write_200(store, run_200)
        # Simulate a stale/duplicate copy of the same run id also living
        # under results/_invalid/ -- must be rejected even though the
        # "normal" copy also exists and would otherwise look usable.
        invalid_dir = tmp_path / "_invalid" / run_200
        invalid_dir.mkdir(parents=True)
        (invalid_dir / "records.jsonl").write_text("")

        with pytest.raises(make_length_pairs.PairDiscoveryError, match="_invalid"):
            make_length_pairs.build_pairs(tmp_path)

    def test_fails_loudly_when_200_run_missing_from_results_dir(self, tmp_path):
        # Nothing written at all -- the hardcoded 200-token run id for
        # sft/en doesn't exist under results_dir.
        with pytest.raises(make_length_pairs.PairDiscoveryError, match="not found directly under"):
            make_length_pairs.build_pairs(tmp_path)
