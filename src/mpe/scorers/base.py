from __future__ import annotations

from abc import ABC, abstractmethod

from mpe.datasets.schema import BenchmarkItem
from mpe.evaluators.schema import RawResponse
from mpe.scorers.schema import ScorerVerdict


class Scorer(ABC):
    """Authoritative Tier-2 scorer for one (item, response) pair.

    Separate interface from Evaluator: an Evaluator generates a candidate
    model's response to a benchmark item; a Scorer independently judges that
    response (e.g. a dedicated safety classifier). Keeping these as two
    interfaces -- rather than folding scoring into the Evaluator -- is what
    lets ExperimentRunner swap scorers (or run without one, falling back to
    Tier-1 rule-based checks only) without touching generation at all.
    """

    scorer_name: str

    @abstractmethod
    def score(self, item: BenchmarkItem, response: RawResponse) -> ScorerVerdict:
        raise NotImplementedError
