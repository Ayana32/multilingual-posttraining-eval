"""End-to-end pipeline test: config -> registry -> loaders -> evaluator ->
parsing -> metrics -> storage, all wired together through ExperimentRunner.

Uses the real checkpoint registry and the real XSTest loader (local files,
no network) plus tiny injected PolyGuard/Belebele loaders (no network) and
MockEvaluator (no model weights) -- this validates the *wiring*, not any
real model's behaviour. See docs/phase1-notes.md.
"""

from mpe.checkpoints.registry import CheckpointRegistry
from mpe.checkpoints.schema import CheckpointStage
from mpe.config.experiment import ExperimentConfig
from mpe.datasets.polyguard import PolyGuardPromptsLoader
from mpe.datasets.xstest import XSTestLoader
from mpe.evaluators.mock import MockEvaluator
from mpe.runner.experiment_runner import ExperimentRunner
from mpe.storage.store import ResultStore

REGISTRY_PATH = "configs/models/olmo3_lineages.yaml"


def test_full_pipeline_produces_expected_record_count(tmp_path, tiny_polyguard_parquet):
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = MockEvaluator()
    store = ResultStore(results_dir=tmp_path)
    loaders = {
        "polyguard_prompts": PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet),
        "xstest": XSTestLoader(),
    }
    runner = ExperimentRunner(registry, evaluator, store, loaders=loaders)

    config = ExperimentConfig(
        name="pipeline_test",
        lineage="instruct",
        stages=[CheckpointStage.BASE, CheckpointStage.RLVR],
        languages=["en", "ko"],
        benchmarks=["polyguard_prompts", "xstest"],
        limit_per_benchmark=3,
        seed=0,
    )

    run_id = runner.run(config)
    records = store.read(run_id)

    # 2 stages x 2 benchmarks x 2 languages x 3 items = 24
    assert len(records) == 24
    assert {r.stage for r in records} == {CheckpointStage.BASE, CheckpointStage.RLVR}
    assert {r.benchmark for r in records} == {"polyguard_prompts", "xstest"}
    assert all(r.run_id == run_id for r in records)
    assert all(r.is_parseable for r in records)  # MockEvaluator never produces empty text


def test_unknown_benchmark_fails_fast(tmp_path):
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = MockEvaluator()
    store = ResultStore(results_dir=tmp_path)
    runner = ExperimentRunner(registry, evaluator, store, loaders={"xstest": XSTestLoader()})

    config = ExperimentConfig(
        name="bad",
        benchmarks=["not_a_real_benchmark"],
        stages=[CheckpointStage.BASE],
        languages=["en"],
        limit_per_benchmark=2,
    )
    import pytest

    with pytest.raises(KeyError, match="Unknown benchmark"):
        runner.run(config)


def test_records_carry_checkpoint_identity_correctly(tmp_path, tiny_polyguard_parquet):
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = MockEvaluator()
    store = ResultStore(results_dir=tmp_path)
    loaders = {"polyguard_prompts": PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)}
    runner = ExperimentRunner(registry, evaluator, store, loaders=loaders)

    config = ExperimentConfig(
        name="identity_test",
        stages=[CheckpointStage.RLVR],
        languages=["en"],
        benchmarks=["polyguard_prompts"],
        limit_per_benchmark=2,
    )
    run_id = runner.run(config)
    records = store.read(run_id)
    assert all(r.hf_repo_id == "allenai/Olmo-3-7B-Instruct" for r in records)
    assert all(r.revision == "main" for r in records)
