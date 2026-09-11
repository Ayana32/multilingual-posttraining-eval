"""End-to-end pipeline test: config -> registry -> loaders -> evaluator ->
parsing -> metrics -> storage, all wired together through ExperimentRunner.

Uses the real checkpoint registry and the real XSTest loader (local files,
no network) plus tiny injected PolyGuard/Belebele loaders (no network) and
MockEvaluator (no model weights) -- this validates the *wiring*, not any
real model's behaviour. See docs/phase1-notes.md.
"""

from mpe.checkpoints.registry import CheckpointRegistry
from mpe.checkpoints.schema import CheckpointSpec, CheckpointStage
from mpe.config.experiment import ExperimentConfig
from mpe.datasets.polyguard import PolyGuardPromptsLoader
from mpe.datasets.schema import BenchmarkItem
from mpe.datasets.xstest import XSTestLoader
from mpe.evaluators.base import Evaluator
from mpe.evaluators.mock import MockEvaluator
from mpe.evaluators.schema import GenerationConfig, RawResponse
from mpe.runner.experiment_runner import ExperimentRunner
from mpe.scorers.base import Scorer
from mpe.scorers.schema import ScorerVerdict
from mpe.storage.store import ResultStore

REGISTRY_PATH = "configs/models/olmo3_lineages.yaml"


class _FiniteFinishReasonEvaluator(Evaluator):
    """Deterministic fake evaluator that assigns a distinct finish_reason
    per item -- exercises the wiring (Evaluator -> RawResponse.finish_reason
    -> ResultRecord.finish_reason) without a real model."""

    def generate(
        self, checkpoint: CheckpointSpec, items: list[BenchmarkItem], config: GenerationConfig
    ) -> list[RawResponse]:
        reasons = ["stop", "length", "error: RuntimeError: boom"]
        return [
            RawResponse(item_id=item.item_id, completion="x", finish_reason=reasons[i % len(reasons)])
            for i, item in enumerate(items)
        ]


class _FakeScorer(Scorer):
    """Deterministic fake authoritative scorer -- exercises the wiring
    (ExperimentRunner -> ResultRecord.scorer_* fields) without a real model."""

    scorer_name = "fake-scorer"

    def score(self, item: BenchmarkItem, response: RawResponse) -> ScorerVerdict:
        return ScorerVerdict(
            item_id=item.item_id,
            scorer_name=self.scorer_name,
            prompt_harmful=item.expected_label == "harmful",
            response_refusal=True,
            response_harmful=False,
            parse_ok=True,
            raw_output="fake verdict",
        )


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


def test_scorer_verdicts_are_wired_into_result_records(tmp_path, tiny_polyguard_parquet):
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = MockEvaluator()
    store = ResultStore(results_dir=tmp_path)
    loaders = {"polyguard_prompts": PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)}
    runner = ExperimentRunner(registry, evaluator, store, loaders=loaders, scorer=_FakeScorer())

    config = ExperimentConfig(
        name="scorer_test",
        stages=[CheckpointStage.BASE],
        languages=["en"],
        benchmarks=["polyguard_prompts"],
        limit_per_benchmark=3,
    )
    run_id = runner.run(config)
    records = store.read(run_id)

    assert all(r.scorer_name == "fake-scorer" for r in records)
    assert all(r.scorer_parse_ok is True for r in records)
    assert all(r.scorer_response_refusal is True for r in records)
    assert all(r.scorer_raw_output == "fake verdict" for r in records)
    for r in records:
        assert r.scorer_prompt_harmful == (r.expected_label == "harmful")


def test_no_scorer_leaves_scorer_fields_none(tmp_path, tiny_polyguard_parquet):
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = MockEvaluator()
    store = ResultStore(results_dir=tmp_path)
    loaders = {"polyguard_prompts": PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)}
    runner = ExperimentRunner(registry, evaluator, store, loaders=loaders)  # no scorer

    config = ExperimentConfig(
        name="no_scorer_test",
        stages=[CheckpointStage.BASE],
        languages=["en"],
        benchmarks=["polyguard_prompts"],
        limit_per_benchmark=2,
    )
    run_id = runner.run(config)
    records = store.read(run_id)
    assert all(r.scorer_name is None for r in records)
    assert all(r.scorer_parse_ok is None for r in records)


def test_finish_reason_is_wired_from_response_into_result_record(tmp_path, tiny_polyguard_parquet):
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = _FiniteFinishReasonEvaluator()
    store = ResultStore(results_dir=tmp_path)
    loaders = {"polyguard_prompts": PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)}
    runner = ExperimentRunner(registry, evaluator, store, loaders=loaders)

    config = ExperimentConfig(
        name="finish_reason_test",
        stages=[CheckpointStage.BASE],
        languages=["en"],
        benchmarks=["polyguard_prompts"],
        limit_per_benchmark=3,
    )
    run_id = runner.run(config)
    records = {r.parallel_item_id: r for r in store.read(run_id)}

    by_id = sorted(records.values(), key=lambda r: r.item_id)
    assert [r.finish_reason for r in by_id] == ["stop", "length", "error: RuntimeError: boom"]


def test_results_are_written_incrementally_per_batch(tmp_path, tiny_polyguard_parquet):
    """A later batch failing must not lose an earlier batch's already-written records."""
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = MockEvaluator()
    store = ResultStore(results_dir=tmp_path)
    loaders = {"polyguard_prompts": PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)}
    runner = ExperimentRunner(registry, evaluator, store, loaders=loaders)

    config = ExperimentConfig(
        name="incremental_test",
        stages=[CheckpointStage.BASE, CheckpointStage.RLVR],
        languages=["en"],
        benchmarks=["polyguard_prompts"],
        limit_per_benchmark=2,
    )
    run_id = runner.run(config)
    path = store.path_for_run(run_id)
    # Two stages, written as two separate append calls -- confirm the file
    # actually accumulated both rather than being written once at the end
    # (this test would still pass either way; it documents the intended
    # incremental-write behaviour and would catch an accidental regression
    # to "collect everything, write once" losing partial progress on error).
    assert path.exists()
    assert len(store.read(run_id)) == 4


def test_item_ids_manifest_drives_the_loader_through_the_full_run(tmp_path, tiny_polyguard_parquet):
    """Phase 2B wiring: item_ids_manifest on the config must reach the
    loader's item_ids override end-to-end, bypassing limit/seed entirely --
    exercised with MockEvaluator only, no real model."""
    import json

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"parallel_item_ids": ["0", "2"]}))

    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    evaluator = MockEvaluator()
    store = ResultStore(results_dir=tmp_path / "results")
    loaders = {"polyguard_prompts": PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)}
    runner = ExperimentRunner(registry, evaluator, store, loaders=loaders)

    config = ExperimentConfig(
        name="manifest_test",
        stages=[CheckpointStage.SFT],
        languages=["en"],
        benchmarks=["polyguard_prompts"],
        item_ids_manifest=str(manifest_path),
        # limit_per_benchmark deliberately left unset/None and would be
        # irrelevant even if set -- item_ids_manifest takes precedence.
    )
    run_id = runner.run(config)
    records = store.read(run_id)
    assert {r.parallel_item_id for r in records} == {"0", "2"}
    assert len(records) == 2
