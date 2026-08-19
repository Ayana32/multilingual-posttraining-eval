from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from mpe.checkpoints.schema import STAGE_ORDER, CheckpointStage
from mpe.evaluators.schema import GenerationConfig


class ExperimentConfig(BaseModel):
    """Everything needed to reproduce one experiment run.

    Deliberately does not validate `lineage`/`benchmarks` against the live
    registries at parse time (that would couple this module to the
    checkpoint/dataset registries just to construct a config object) --
    ExperimentRunner does that lookup and fails fast with a clear error at
    run() time instead.
    """

    name: str
    lineage: str = "instruct"
    stages: list[CheckpointStage] = Field(default_factory=lambda: list(STAGE_ORDER))
    languages: list[str] = Field(default_factory=lambda: ["en", "ko"])
    benchmarks: list[str]
    limit_per_benchmark: int | None = None
    seed: int = 0
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @field_validator("stages")
    @classmethod
    def _stages_non_empty(cls, v: list[CheckpointStage]) -> list[CheckpointStage]:
        if not v:
            raise ValueError("stages must not be empty")
        return v

    @field_validator("languages")
    @classmethod
    def _languages_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("languages must not be empty")
        return v

    @field_validator("benchmarks")
    @classmethod
    def _benchmarks_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("benchmarks must not be empty")
        return v

    @field_validator("limit_per_benchmark")
    @classmethod
    def _limit_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("limit_per_benchmark must be positive if set")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)
