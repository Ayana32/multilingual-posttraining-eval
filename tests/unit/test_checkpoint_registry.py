from pathlib import Path

import pytest
import yaml

from mpe.checkpoints import (
    COMPARABLE_TRAJECTORY_STAGES,
    DESCRIPTIVE_REFERENCE_STAGES,
    CheckpointRegistry,
    CheckpointStage,
)

REGISTRY_PATH = Path("configs/models/olmo3_lineages.yaml")


def test_base_is_descriptive_reference_not_comparable_trajectory():
    assert CheckpointStage.BASE in DESCRIPTIVE_REFERENCE_STAGES
    assert CheckpointStage.BASE not in COMPARABLE_TRAJECTORY_STAGES


def test_comparable_trajectory_is_exactly_sft_dpo_rlvr_in_order():
    assert COMPARABLE_TRAJECTORY_STAGES == [
        CheckpointStage.SFT,
        CheckpointStage.DPO,
        CheckpointStage.RLVR,
    ]


def test_trajectory_and_reference_stages_are_disjoint_and_cover_all_stages():
    assert set(COMPARABLE_TRAJECTORY_STAGES) & set(DESCRIPTIVE_REFERENCE_STAGES) == set()
    assert set(COMPARABLE_TRAJECTORY_STAGES) | set(DESCRIPTIVE_REFERENCE_STAGES) == set(CheckpointStage)


def test_loads_real_registry_file():
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    assert "instruct" in registry.lineage_names
    assert "think" in registry.lineage_names


def test_mvp_lineage_has_all_four_stages():
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    lineage = registry.mvp_lineage()
    assert lineage.name == "instruct"
    assert set(lineage.stage_names) == {
        CheckpointStage.BASE,
        CheckpointStage.SFT,
        CheckpointStage.DPO,
        CheckpointStage.RLVR,
    }


def test_rlvr_checkpoint_matches_verified_repo_id():
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    checkpoint = registry.get_checkpoint("instruct", CheckpointStage.RLVR)
    assert checkpoint.hf_repo_id == "allenai/Olmo-3-7B-Instruct"
    assert checkpoint.base_model == "allenai/Olmo-3-7B-Instruct-DPO"


def test_unknown_lineage_raises_with_helpful_message():
    registry = CheckpointRegistry.from_yaml(REGISTRY_PATH)
    with pytest.raises(KeyError, match="Unknown lineage"):
        registry.get_lineage("does-not-exist")


def test_unknown_stage_raises(tmp_path):
    minimal = {
        "selected_for_mvp": "toy",
        "lineages": {
            "toy": {
                "architecture": "Toy",
                "parameters": "0",
                "license": "mit",
                "status": "test",
                "stages": [{"stage": "base", "hf_repo_id": "toy/base"}],
            }
        },
    }
    path = tmp_path / "toy_registry.yaml"
    path.write_text(yaml.dump(minimal))
    registry = CheckpointRegistry.from_yaml(path)
    lineage = registry.get_lineage("toy")
    with pytest.raises(KeyError):
        lineage.get_stage(CheckpointStage.RLVR)


def test_malformed_registry_missing_stages_field_raises(tmp_path):
    malformed = {"lineages": {"bad": {"architecture": "X", "parameters": "0", "license": "mit", "status": "x"}}}
    path = tmp_path / "malformed.yaml"
    path.write_text(yaml.dump(malformed))
    with pytest.raises(Exception):
        CheckpointRegistry.from_yaml(path)
