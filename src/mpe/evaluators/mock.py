"""Deterministic, dependency-free stand-in for a real model.

Exists so the runner/parsing/metrics/storage pipeline can be exercised
end-to-end -- in unit tests, in CI (once Phase 3 adds it), and in the Phase 1
smoke-test script -- without torch, a GPU, or a Hub download. This is
infrastructure-validation only. It has no access to any real model's actual
behaviour and must never be cited as evidence for a research finding.

Default behaviour is stage-neutral: refusal/compliance noise is drawn from
the same distribution regardless of which post-training stage is passed in,
so a default smoke-test run cannot be mistaken for a preliminary result
showing a stage-over-stage trend. A synthetic, clearly-named, opt-in bias
(TEST_ONLY_SYNTHETIC_STAGE_BIAS) exists purely to give a demo something
non-flat to display when that's explicitly wanted -- it is off by default
and warns loudly when enabled.
"""

from __future__ import annotations

import warnings
import hashlib

from mpe.checkpoints.schema import CheckpointSpec
from mpe.datasets.schema import BenchmarkItem, TaskType
from mpe.evaluators.base import Evaluator
from mpe.evaluators.schema import GenerationConfig, RawResponse

_REFUSAL_TEXT = {
    "en": "I'm sorry, but I can't help with that request.",
    "ko": "죄송하지만 그 요청은 도와드릴 수 없습니다.",
}
_COMPLIANCE_TEXT = {
    "en": "Sure, here's an answer to your question about: {snippet}",
    "ko": "네, 다음 질문에 대한 답변입니다: {snippet}",
}

TEST_ONLY_SYNTHETIC_STAGE_BIAS: dict[str, float] = {
    "base": -0.05,
    "sft": 0.0,
    "dpo": 0.05,
    "rlvr": 0.10,
}
"""NOT derived from any real OLMo behaviour. Purely a knob to make a demo run
show a non-flat stage trend on request. OFF unless MockEvaluator is
constructed with use_test_only_synthetic_stage_bias=True -- see that
docstring. Never wire this into anything that produces or resembles a
research result."""


def _roll(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockEvaluator(Evaluator):
    def __init__(self, noise_rate: float = 0.1, use_test_only_synthetic_stage_bias: bool = False):
        """
        Args:
            noise_rate: probability the mock "flips" the ground-truth-implied
                behaviour, applied identically across every stage by default.
            use_test_only_synthetic_stage_bias: if True, applies
                TEST_ONLY_SYNTHETIC_STAGE_BIAS on top of noise_rate so later
                stages refuse slightly more often. Exists only to make a demo
                visually interesting. Emits a RuntimeWarning when enabled so
                it cannot be turned on silently. Default False.
        """
        self.noise_rate = noise_rate
        self.use_test_only_synthetic_stage_bias = use_test_only_synthetic_stage_bias
        if use_test_only_synthetic_stage_bias:
            warnings.warn(
                "MockEvaluator(use_test_only_synthetic_stage_bias=True): output "
                "will show a FAKE stage-over-stage trend that is not derived "
                "from any real model. Do not report this as a research "
                "finding or preliminary result.",
                stacklevel=2,
            )

    def generate(
        self,
        checkpoint: CheckpointSpec,
        items: list[BenchmarkItem],
        config: GenerationConfig,
    ) -> list[RawResponse]:
        responses = []
        for item in items:
            if item.task_type == TaskType.MULTIPLE_CHOICE:
                completion = self._mc_completion(item, checkpoint, config.seed)
            else:
                completion = self._refusal_completion(item, checkpoint, config.seed)
            responses.append(
                RawResponse(item_id=item.item_id, completion=completion, finish_reason="stop")
            )
        return responses

    def _effective_noise(self, item: BenchmarkItem, checkpoint: CheckpointSpec) -> float:
        ground_truth_refuse = item.expected_label in {"harmful", "unsafe"}
        bias = 0.0
        if self.use_test_only_synthetic_stage_bias:
            bias = TEST_ONLY_SYNTHETIC_STAGE_BIAS.get(checkpoint.stage.value, 0.0)
        raw = self.noise_rate - bias if ground_truth_refuse else self.noise_rate + bias
        return max(0.0, min(1.0, raw))

    def _should_refuse(self, item: BenchmarkItem, checkpoint: CheckpointSpec, seed: int) -> bool:
        ground_truth_refuse = item.expected_label in {"harmful", "unsafe"}
        effective_noise = self._effective_noise(item, checkpoint)
        flip = _roll(item.item_id, checkpoint.stage.value, str(seed)) < effective_noise
        return (not ground_truth_refuse) if flip else ground_truth_refuse

    def _refusal_completion(self, item: BenchmarkItem, checkpoint: CheckpointSpec, seed: int) -> str:
        lang = item.language if item.language in _REFUSAL_TEXT else "en"
        if self._should_refuse(item, checkpoint, seed):
            return _REFUSAL_TEXT[lang]
        return _COMPLIANCE_TEXT[lang].format(snippet=item.prompt[:40])

    def _mc_completion(self, item: BenchmarkItem, checkpoint: CheckpointSpec, seed: int) -> str:
        correct = item.correct_choice_index if item.correct_choice_index is not None else 0
        num_choices = len(item.choices) if item.choices else 4
        roll = _roll(item.item_id, checkpoint.stage.value, str(seed), "mc")
        chosen = correct if roll < 0.9 else (correct + 1) % num_choices
        return f"Answer: {chosen + 1}"
