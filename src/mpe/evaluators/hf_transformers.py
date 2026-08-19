"""Real evaluator: loads a checkpoint via transformers and generates.

This is the evaluator Phase 2's authoritative cloud-GPU runs will use (see
docs/phase0-findings.md section 8). It has NOT been exercised against real
weights in this environment -- the dev machine has no confirmed GPU/torch
setup for a 7B model (see docs/phase0-findings.md section 8) -- so torch and
transformers are imported lazily inside __init__ rather than at module load
time. This keeps `import mpe` and every other evaluator/loader/metric usable
on a machine that never installs the `hf` extra, and makes the missing-
dependency failure happen at the one call site that actually needs it,
with a clear message, instead of an opaque import error somewhere unrelated.
"""

from __future__ import annotations

from mpe.checkpoints.schema import CheckpointSpec
from mpe.datasets.schema import BenchmarkItem
from mpe.evaluators.base import Evaluator
from mpe.evaluators.schema import GenerationConfig, RawResponse


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
                torch_dtype=checkpoint.dtype,
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

        responses = []
        for item in items:
            messages = []
            if config.system_prompt_override is not None:
                messages.append({"role": "system", "content": config.system_prompt_override})
            messages.append({"role": "user", "content": item.prompt})

            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=config.temperature if config.temperature > 0 else None,
            )
            completion = tokenizer.decode(
                output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True
            )
            responses.append(RawResponse(item_id=item.item_id, completion=completion))
        return responses
