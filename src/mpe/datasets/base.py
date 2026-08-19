"""DatasetLoader interface shared by every benchmark loader."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Hashable, Sequence, TypeVar

from mpe.datasets.schema import BenchmarkItem

T = TypeVar("T", bound=Hashable)


class DatasetLoader(ABC):
    """Loads BenchmarkItems for one benchmark, one language at a time.

    Implementations must guarantee that calling load() twice with the same
    (limit, seed) but different languages returns *parallel* items -- i.e.
    the same set of parallel_item_id values -- wherever the underlying data
    supports that. This is what makes cross-lingual comparison well-defined
    rather than comparing two independently-drawn samples.
    """

    benchmark_name: str

    @abstractmethod
    def load(
        self, language: str, limit: int | None = None, seed: int = 0
    ) -> list[BenchmarkItem]:
        raise NotImplementedError


def deterministic_sample(keys: Sequence[T], limit: int | None, seed: int) -> list[T]:
    """Seeded, order-independent subsample of `keys`.

    Raises rather than silently truncating when `limit` exceeds what's
    available -- an eval framework that quietly returns fewer items than an
    experiment config asked for would corrupt the experiment's assumed
    sample size without anyone noticing. Sorted before sampling so the same
    `keys` passed in any order yields the same result for a given seed.
    """
    ordered = sorted(keys)  # type: ignore[type-var]
    if limit is None:
        return ordered
    if limit > len(ordered):
        raise ValueError(
            f"Requested limit={limit} exceeds the {len(ordered)} items available"
        )
    return sorted(random.Random(seed).sample(ordered, limit))
