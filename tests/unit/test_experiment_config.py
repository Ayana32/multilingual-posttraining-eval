import pytest
import yaml

from mpe.checkpoints.schema import CheckpointStage
from mpe.config.experiment import ExperimentConfig


def test_loads_real_smoke_test_config():
    config = ExperimentConfig.from_yaml("configs/experiments/smoke_test.yaml")
    assert config.name == "smoke_test"
    assert config.stages == [
        CheckpointStage.BASE,
        CheckpointStage.SFT,
        CheckpointStage.DPO,
        CheckpointStage.RLVR,
    ]
    assert config.languages == ["en", "ko"]
    assert config.limit_per_benchmark == 4


def test_defaults_cover_all_four_stages_and_both_languages():
    config = ExperimentConfig(name="x", benchmarks=["polyguard_prompts"])
    assert len(config.stages) == 4
    assert config.languages == ["en", "ko"]


def test_empty_benchmarks_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", benchmarks=[])


def test_empty_stages_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", benchmarks=["polyguard_prompts"], stages=[])


def test_non_positive_limit_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", benchmarks=["polyguard_prompts"], limit_per_benchmark=0)


def test_invalid_stage_name_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", benchmarks=["polyguard_prompts"], stages=["not-a-stage"])


def test_malformed_yaml_missing_required_field_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump({"lineage": "instruct"}))  # missing required 'name' and 'benchmarks'
    with pytest.raises(Exception):
        ExperimentConfig.from_yaml(path)


def test_item_ids_manifest_defaults_to_none():
    config = ExperimentConfig(name="x", benchmarks=["polyguard_prompts"])
    assert config.item_ids_manifest is None


def test_item_ids_manifest_can_be_set():
    config = ExperimentConfig(
        name="x", benchmarks=["polyguard_prompts"], item_ids_manifest="configs/samples/foo.json"
    )
    assert config.item_ids_manifest == "configs/samples/foo.json"


@pytest.mark.parametrize(
    "path",
    [
        "configs/experiments/phase2b_sft_en_polyguard.yaml",
        "configs/experiments/phase2b_sft_ko_polyguard.yaml",
        "configs/experiments/phase2b_dpo_en_polyguard.yaml",
        "configs/experiments/phase2b_dpo_ko_polyguard.yaml",
        "configs/experiments/phase2b_rlvr_en_polyguard.yaml",
        "configs/experiments/phase2b_rlvr_ko_polyguard.yaml",
    ],
)
def test_phase2b_configs_reference_the_shared_manifest(path):
    config = ExperimentConfig.from_yaml(path)
    assert config.item_ids_manifest == "configs/samples/phase2b_polyguard_en_ko_200.json"
    assert config.benchmarks == ["polyguard_prompts"]
    assert len(config.stages) == 1
    assert len(config.languages) == 1
