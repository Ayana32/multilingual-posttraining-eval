from datetime import datetime, timezone

import pytest

from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _record(run_id: str = "run1", item_id: str = "item1") -> ResultRecord:
    return ResultRecord(
        run_id=run_id,
        experiment_name="exp",
        lineage="instruct",
        stage=CheckpointStage.SFT,
        hf_repo_id="allenai/Olmo-3-7B-Instruct-SFT",
        revision="main",
        language="en",
        benchmark="polyguard_prompts",
        item_id=item_id,
        parallel_item_id="1",
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="prompt text",
        completion="completion text",
        is_parseable=True,
        language_match=True,
        mc_correct=None,
        refusal_label="refusal",
        behavior_matches_expected=True,
        expected_label="harmful",
        generation_config=GenerationConfig(),
        created_at=datetime.now(timezone.utc),
    )


def test_write_then_read_round_trip(tmp_path):
    store = ResultStore(results_dir=tmp_path)
    records = [_record(item_id="a"), _record(item_id="b")]
    store.write(records)

    read_back = store.read("run1")
    assert len(read_back) == 2
    assert {r.item_id for r in read_back} == {"a", "b"}
    assert read_back[0].stage == CheckpointStage.SFT


def test_write_is_append_not_overwrite(tmp_path):
    store = ResultStore(results_dir=tmp_path)
    store.write([_record(item_id="a")])
    store.write([_record(item_id="b")])
    assert len(store.read("run1")) == 2


def test_read_missing_run_returns_empty_list(tmp_path):
    store = ResultStore(results_dir=tmp_path)
    assert store.read("does-not-exist") == []


def test_write_empty_list_raises(tmp_path):
    store = ResultStore(results_dir=tmp_path)
    with pytest.raises(ValueError):
        store.write([])


def test_write_mixed_run_ids_raises(tmp_path):
    store = ResultStore(results_dir=tmp_path)
    with pytest.raises(ValueError, match="multiple runs"):
        store.write([_record(run_id="run1"), _record(run_id="run2")])


def test_list_runs(tmp_path):
    store = ResultStore(results_dir=tmp_path)
    store.write([_record(run_id="run_a")])
    store.write([_record(run_id="run_b")])
    assert store.list_runs() == ["run_a", "run_b"]
