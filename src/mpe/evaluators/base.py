from __future__ import annotations

from abc import ABC, abstractmethod

from mpe.checkpoints.schema import CheckpointSpec
from mpe.datasets.schema import BenchmarkItem
from mpe.evaluators.schema import GenerationConfig, RawResponse


class Evaluator(ABC):
    """Generates raw model output for a batch of items against one checkpoint.

    Deliberately the only interface boundary between "how do we get text out
    of a model" and everything downstream (parsing, metrics, storage). Swap
    MockEvaluator for HFTransformersEvaluator (or a future vLLM/cloud-API
    evaluator) without touching any other module.
    """

    @abstractmethod
    def generate(
        self,
        checkpoint: CheckpointSpec,
        items: list[BenchmarkItem],
        config: GenerationConfig,
    ) -> list[RawResponse]:
        raise NotImplementedError
