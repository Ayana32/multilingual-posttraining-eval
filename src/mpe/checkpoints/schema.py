"""Typed schema for OLMo checkpoint stages and lineages.

Mirrors the structure of configs/models/olmo3_lineages.yaml, which is the
Phase 0/1 verified source of truth (see docs/phase0-findings.md section 1
and docs/phase1-notes.md section 5 for how each field was confirmed).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CheckpointStage(str, Enum):
    BASE = "base"
    SFT = "sft"
    DPO = "dpo"
    RLVR = "rlvr"


# Canonical stage order for a post-training trajectory. Used anywhere we need
# to walk "the next stage" (e.g. computing stage-over-stage deltas).
STAGE_ORDER: list[CheckpointStage] = [
    CheckpointStage.BASE,
    CheckpointStage.SFT,
    CheckpointStage.DPO,
    CheckpointStage.RLVR,
]


class CheckpointSpec(BaseModel):
    """One verified checkpoint: a single stage within a lineage."""

    stage: CheckpointStage
    hf_repo_id: str
    revision: str = "main"
    dtype: str = "bfloat16"
    base_model: str | None = None
    role: str | None = None
    verified_via: str | None = None
    known_issue: str | None = None


class LineageSpec(BaseModel):
    """A full Base->SFT->DPO->RLVR trajectory for one model family variant."""

    name: str
    architecture: str
    parameters: str
    license: str
    status: str
    chat_template_file: str | None = None
    chat_template_notes: str | None = None
    rejection_reason: str | None = None
    stages: list[CheckpointSpec]

    def get_stage(self, stage: CheckpointStage) -> CheckpointSpec:
        for spec in self.stages:
            if spec.stage == stage:
                return spec
        raise KeyError(f"Lineage '{self.name}' has no stage '{stage.value}'")

    @property
    def stage_names(self) -> list[CheckpointStage]:
        return [s.stage for s in self.stages]
