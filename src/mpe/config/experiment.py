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
    item_ids_manifest: str | None = None
    """Path to a checked-in JSON manifest of frozen parallel_item_ids (see
    mpe.datasets.manifest.load_id_manifest and configs/samples/) -- an
    alternative to limit_per_benchmark + seed for pinning an exact,
    auditable sample rather than re-deriving one via seeded sampling on
    every run. When set, ExperimentRunner passes the manifest's ids to
    any loader that supports an `item_ids` override (currently
    PolyGuardPromptsLoader only); limit_per_benchmark and seed are then
    ignored for that benchmark. Not validated here (no file I/O at
    construction time, consistent with this class's existing policy of
    deferring registry/dataset lookups to ExperimentRunner.run()) --
    a missing or malformed manifest fails fast at run() time instead.
    """
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
