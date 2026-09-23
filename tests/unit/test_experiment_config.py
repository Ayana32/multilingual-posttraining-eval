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
def test_phase2b_configs_use_1024_max_new_tokens_for_the_truncation_ablation(path):
    """Phase 2A/2B's original 200-token pilots showed a heavily asymmetric
    truncation pattern (especially KO) confounding EN/KO interpretation;
    Phase 2B was updated to max_new_tokens=1024 to remove that confound
    before real execution -- see the len1024 ablation configs below."""
    config = ExperimentConfig.from_yaml(path)
    assert config.generation.max_new_tokens == 1024


LEN1024_PILOT_PAIRS = [
    ("sft", "en"),
    ("sft", "ko"),
    ("dpo", "en"),
    ("dpo", "ko"),
    ("rlvr", "en"),
    ("rlvr", "ko"),
]


@pytest.mark.parametrize("stage, lang", LEN1024_PILOT_PAIRS)
def test_len1024_pilot_config_parses(stage, lang):
    path = f"configs/experiments/pilot_{stage}_{lang}_polyguard_len1024.yaml"
    config = ExperimentConfig.from_yaml(path)
    assert config.name == f"pilot_{stage}_{lang}_polyguard_len1024"
    assert config.generation.max_new_tokens == 1024


@pytest.mark.parametrize("stage, lang", LEN1024_PILOT_PAIRS)
def test_len1024_pilot_config_matches_source_except_name_and_max_new_tokens(stage, lang):
    """The only intended differences between a len1024 ablation config and
    its source pilot config are `name` (with `_len1024` appended) and
    `generation.max_new_tokens` (200 -> 1024) -- checkpoint, language,
    benchmark, sample size/seed, generation seed, temperature, and every
    other field must be byte-for-byte identical once those two are
    normalized away."""
    source_path = f"configs/experiments/pilot_{stage}_{lang}_polyguard.yaml"
    len1024_path = f"configs/experiments/pilot_{stage}_{lang}_polyguard_len1024.yaml"

    source_dump = ExperimentConfig.from_yaml(source_path).model_dump()
    len1024_dump = ExperimentConfig.from_yaml(len1024_path).model_dump()

    assert len1024_dump.pop("name") == source_dump.pop("name") + "_len1024"

    assert source_dump["generation"]["max_new_tokens"] == 200
    assert len1024_dump["generation"]["max_new_tokens"] == 1024
    source_dump["generation"].pop("max_new_tokens")
    len1024_dump["generation"].pop("max_new_tokens")

    assert source_dump == len1024_dump


@pytest.mark.parametrize(
    "path",
    [
        "configs/experiments/pilot_sft_en_polyguard.yaml",
        "configs/experiments/pilot_sft_ko_polyguard.yaml",
        "configs/experiments/pilot_dpo_en_polyguard.yaml",
        "configs/experiments/pilot_dpo_ko_polyguard.yaml",
        "configs/experiments/pilot_rlvr_en_polyguard.yaml",
        "configs/experiments/pilot_rlvr_ko_polyguard.yaml",
    ],
)
def test_original_200_token_pilot_configs_are_unchanged_by_the_len1024_ablation(path):
    config = ExperimentConfig.from_yaml(path)
    assert config.generation.max_new_tokens == 200
    assert not config.name.endswith("_len1024")
