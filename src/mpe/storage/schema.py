from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig


class ResultRecord(BaseModel):
    """One scored (checkpoint, language, item) observation.

    This is the only artifact anything downstream of the runner is allowed
    to read from -- the eventual API/tools/agent layers (Phase 4/5) query
    ResultStore, never re-run generation or re-score. Keeping every field
    that identifies *how* this record was produced (revision, generation
    config, run_id) on the record itself is what makes a single row
    independently interpretable later, without needing to cross-reference
    which config produced which run.
    """

    run_id: str
    experiment_name: str
    lineage: str
    stage: CheckpointStage
    hf_repo_id: str
    revision: str

    language: str
    benchmark: str
    item_id: str
    parallel_item_id: str
    task_type: TaskType

    prompt: str
    completion: str

    is_parseable: bool
    language_match: bool | None
    mc_correct: bool | None
    refusal_label: str | None
    behavior_matches_expected: bool | None
    expected_label: str | None

    generation_config: GenerationConfig
    created_at: datetime
