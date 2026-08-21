"""Real evaluator: loads a checkpoint via transformers and generates.

This is the evaluator Phase 2's authoritative cloud-GPU runs will use (see
docs/phase0-findings.md section 8). torch and transformers are imported
lazily inside __init__ rather than at module load time, so `import mpe` and
every other evaluator/loader/metric stays usable on a machine that never
installs the 'hf' extra, and the missing-dependency failure happens at the
one call site that actually needs it, with a clear message.

Chat-template policy (see docs/phase2a-pilot.md section "chat template
validation" for the full verification):
  - SFT/DPO/RLVR (Instruct lineage) all carry a chat template that, when no
    system message is supplied, injects each checkpoint's own default
    system prompt identically for our use (no-tools) case -- confirmed by
    diffing all three templates directly, not assumed. This evaluator
    therefore passes NO explicit system message by default
    (system_prompt_override=None), deliberately relying on that shared
    default so the system-prompt condition is held constant across those
    three stages without this code needing to hardcode or duplicate it.
  - Base has NO chat template at all (confirmed: no chat_template.jinja file,
    no chat_template key in tokenizer_config.json) -- it is a pretrain-only
    checkpoint with no instruction-following behaviour to invoke a template
    for. This evaluator detects that and falls back to raw prompt
    continuation (the prompt text tokenized directly, no chat wrapper, no
    system prompt). This is standard practice for base-LM evaluation, not a
    workaround -- but it is a genuine, unavoidable asymmetry with the other
    three stages and must never be silently treated as "the same protocol."
"""

from __future__ import annotations

from mpe.checkpoints.schema import CheckpointSpec
from mpe.datasets.schema import BenchmarkItem
from mpe.evaluators.base import Evaluator
from mpe.evaluators.schema import GenerationConfig, RawResponse


def supports_chat_template(tokenizer) -> bool:
    """True if this tokenizer has a usable chat template.

    A plain function (not a method) so it's unit-testable against a fake
    duck-typed tokenizer object without loading any real model.
    """
    return getattr(tokenizer, "chat_template", None) is not None


def build_model_input(tokenizer, item: BenchmarkItem, config: GenerationConfig):
    """Returns tokenizer output (input_ids etc.) for one item.

    Chat-templated path for checkpoints that have a template; raw
    continuation for those that don't (Base). Pulled out of generate() so
    the branch decision is independently testable with a fake tokenizer.
    """
    if supports_chat_template(tokenizer):
        messages = []
        if config.system_prompt_override is not None:
            messages.append({"role": "system", "content": config.system_prompt_override})
        messages.append({"role": "user", "content": item.prompt})
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    return tokenizer(item.prompt, return_tensors="pt").input_ids


class HFTransformersEvaluator(Evaluator):
    def __init__(self, device: str = "auto"):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "HFTransformersEvaluator requires the 'hf' extra "
                "(pip install -e '.[hf]') -- torch/transformers are not "
                "installed. This is expected on the local dev machine; see "
                "docs/phase0-findings.md section 8 for the cloud-GPU setup "
                "these real runs are meant to use."
            ) from exc
        self.device = device
        self._model_cache: dict[tuple[str, str], object] = {}

    def _load(self, checkpoint: CheckpointSpec):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        key = (checkpoint.hf_repo_id, checkpoint.revision)
        if key not in self._model_cache:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint.hf_repo_id, revision=checkpoint.revision
            )
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint.hf_repo_id,
                revision=checkpoint.revision,
                dtype=checkpoint.dtype,  # `torch_dtype` was renamed `dtype` in transformers 5.x
                device_map=self.device,
            )
            self._model_cache[key] = (tokenizer, model)
        return self._model_cache[key]

    def generate(
        self,
        checkpoint: CheckpointSpec,
        items: list[BenchmarkItem],
        config: GenerationConfig,
    ) -> list[RawResponse]:
        import torch

        tokenizer, model = self._load(checkpoint)
        torch.manual_seed(config.seed)

        protocol = "chat_template" if supports_chat_template(tokenizer) else "raw_continuation"

        responses = []
        for item in items:
            try:
                input_ids = build_model_input(tokenizer, item, config).to(model.device)
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.temperature > 0,
                    temperature=config.temperature if config.temperature > 0 else None,
                )
                new_tokens = output_ids[0][input_ids.shape[-1] :]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                finish_reason = "length" if len(new_tokens) >= config.max_new_tokens else "stop"
                responses.append(
                    RawResponse(
                        item_id=item.item_id,
                        completion=completion,
                        finish_reason=finish_reason,
                        generation_protocol=protocol,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # One bad item (a tokenization edge case, an OOM on an
                # unusually long prompt) must not lose every other item's
                # already-computed result -- see ExperimentRunner's
                # per-batch incremental write for the other half of this.
                responses.append(
                    RawResponse(
                        item_id=item.item_id,
                        completion="",
                        finish_reason=f"error: {type(exc).__name__}: {exc}",
                        generation_protocol=protocol,
                    )
                )
        return responses
