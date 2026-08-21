from __future__ import annotations

from pydantic import BaseModel


class GenerationConfig(BaseModel):
    temperature: float = 0.0
    max_new_tokens: int = 256
    seed: int = 0
    system_prompt_override: str | None = None
    """If None, the checkpoint's default chat template system prompt is used
    as-is (for the Instruct lineage this is AI2's default "helpful
    function-calling AI assistant" prompt -- see
    configs/models/olmo3_lineages.yaml). Must be held constant across an
    entire experiment's checkpoints/languages; ExperimentRunner enforces
    this by reading it once per experiment, not per generation call.
    """


class RawResponse(BaseModel):
    item_id: str
    completion: str
    finish_reason: str = "stop"
    latency_ms: float | None = None
    generation_protocol: str | None = None
    """"chat_template" | "raw_continuation" | None. Set by the Evaluator that
    actually produced this response, based on what it really did (e.g.
    HFTransformersEvaluator sets this from supports_chat_template()), not
    inferred later from the checkpoint's stage name. None means "not
    modeled" (e.g. MockEvaluator, which does no real templating) -- treat
    None the same as "unknown," never as "chat_template" by default.
    """
