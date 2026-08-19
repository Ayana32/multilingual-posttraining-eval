from __future__ import annotations

from pydantic import BaseModel


class ScoredResult(BaseModel):
    item_id: str
    is_parseable: bool
    language_match: bool | None = None
    mc_correct: bool | None = None
    """Belebele-KO competence-control metric: exact-match correctness."""
    refusal_label: str | None = None
    """"refusal" | "compliance", from the Tier-1 rule-based classifier."""
    behavior_matches_expected: bool | None = None
    """The Tier-2 safety signal: does the observed refusal/compliance match
    what the benchmark's label says the desired behaviour is (refuse on
    harmful/unsafe items, comply on unharmful/safe ones)? Per
    docs/phase0-findings.md section 5's gating rule, this must be
    interpreted alongside mc_correct (the competence control) for the same
    checkpoint/language before being reported as a "safety" finding.
    """
