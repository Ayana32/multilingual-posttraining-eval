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
    finish_reason: str = "stop"
    """Mirrors RawResponse.finish_reason ("stop" | "length" | "error: ...").
    Added after the 50-item Base-English pilot ran without it, which is why
    that run's records can't be checked for exact truncation counts -- see
    docs/phase2a-pilot.md. Defaults to "stop" (RawResponse's own default) so
    older stored records without this key still parse instead of failing.
    """
    generation_protocol: str | None = None
    """"chat_template" | "raw_continuation" | None -- what the evaluator
    actually did to build the prompt, not derived from `stage`. This is the
    field any future cross-stage comparison must check before treating two
    rows' stages as directly comparable: a Base ("raw_continuation") row and
    an SFT/DPO/RLVR ("chat_template") row differ in generation_protocol and
    must not be diffed as a clean stage-over-stage effect. See
    mpe.checkpoints.schema.COMPARABLE_TRAJECTORY_STAGES.
    """

    is_parseable: bool
    language_match: bool | None
    mc_correct: bool | None
    refusal_label: str | None
    behavior_matches_expected: bool | None
    expected_label: str | None

    scorer_name: str | None = None
    scorer_prompt_harmful: bool | None = None
    scorer_response_refusal: bool | None = None
    scorer_response_harmful: bool | None = None
    scorer_parse_ok: bool | None = None
    """None = no scorer was run for this record (Tier-1 rule-based fields
    above are all that's available). False = a scorer ran but this
    project's parser could not confidently read its output -- see
    mpe.scorers.polyguard. True = the scorer_* fields above are trustworthy.
    """
    scorer_raw_output: str | None = None
    """The scorer's literal, unparsed text output. Stored even when
    scorer_parse_ok is True -- the parsed booleans above are only as
    trustworthy as PolyGuardScorer.output_format_validated says they are
    (see mpe.scorers.polyguard), and this field is what a human actually
    reads to make that call. Without it, scorer_parse_ok=True is just
    another unverified claim."""

    generation_config: GenerationConfig
    created_at: datetime
