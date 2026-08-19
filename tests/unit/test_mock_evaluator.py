import pytest

from mpe.checkpoints.schema import CheckpointSpec, CheckpointStage
from mpe.datasets.schema import BenchmarkItem, TaskType
from mpe.evaluators.mock import TEST_ONLY_SYNTHETIC_STAGE_BIAS, MockEvaluator
from mpe.evaluators.schema import GenerationConfig

_STAGES = [CheckpointStage.BASE, CheckpointStage.SFT, CheckpointStage.DPO, CheckpointStage.RLVR]


def _checkpoint(stage: CheckpointStage) -> CheckpointSpec:
    return CheckpointSpec(stage=stage, hf_repo_id=f"test/{stage.value}")


def _harmful_item(item_id: str = "x") -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        benchmark="test",
        language="en",
        parallel_item_id="1",
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="p",
        expected_label="harmful",
    )


class TestDefaultIsStageNeutral:
    def test_effective_noise_identical_across_all_stages_by_default(self):
        evaluator = MockEvaluator(noise_rate=0.2)
        item = _harmful_item()
        noises = {
            stage: evaluator._effective_noise(item, _checkpoint(stage)) for stage in _STAGES
        }
        assert len(set(noises.values())) == 1
        assert noises[CheckpointStage.BASE] == 0.2

    def test_bias_dict_not_applied_when_flag_off(self):
        evaluator = MockEvaluator()
        assert evaluator.use_test_only_synthetic_stage_bias is False

    def test_no_warning_by_default(self, recwarn):
        MockEvaluator()
        assert len(recwarn) == 0


class TestOptInSyntheticBias:
    def test_bias_changes_effective_noise_per_stage(self):
        evaluator = MockEvaluator(noise_rate=0.2, use_test_only_synthetic_stage_bias=True)
        item = _harmful_item()
        noises = {
            stage: evaluator._effective_noise(item, _checkpoint(stage)) for stage in _STAGES
        }
        assert len(set(noises.values())) > 1
        for stage in _STAGES:
            expected = 0.2 - TEST_ONLY_SYNTHETIC_STAGE_BIAS[stage.value]
            assert noises[stage] == pytest.approx(max(0.0, min(1.0, expected)))

    def test_enabling_bias_emits_warning(self):
        with pytest.warns(UserWarning, match="FAKE stage-over-stage trend"):
            MockEvaluator(use_test_only_synthetic_stage_bias=True)


class TestGenerateIsDeterministic:
    def test_same_inputs_same_output(self):
        evaluator = MockEvaluator()
        item = _harmful_item()
        config = GenerationConfig(seed=7)
        r1 = evaluator.generate(_checkpoint(CheckpointStage.SFT), [item], config)
        r2 = evaluator.generate(_checkpoint(CheckpointStage.SFT), [item], config)
        assert r1[0].completion == r2[0].completion

    def test_never_produces_empty_completion(self):
        evaluator = MockEvaluator()
        item = _harmful_item()
        responses = evaluator.generate(
            _checkpoint(CheckpointStage.BASE), [item], GenerationConfig()
        )
        assert responses[0].completion.strip() != ""
