from __future__ import annotations

from pydantic import BaseModel


class ScorerVerdict(BaseModel):
    """Output of an authoritative (benchmark-native) scorer for one item.

    Distinct from ParsedOutput/ScoredResult (mpe.parsing, mpe.metrics), which
    are the Tier-1 rule-based checks. A Scorer produces the Tier-2 label that
    is actually meant to be reported -- see docs/phase0-findings.md section 6
    and docs/phase1-review.md section 3 for why PolyGuardPrompts specifically
    should not be scored by the rule-based parser alone.
    """

    item_id: str
    scorer_name: str
    prompt_harmful: bool | None = None
    response_refusal: bool | None = None
    response_harmful: bool | None = None
    parse_ok: bool
    """False if the scorer produced text but this project's parser could not
    confidently extract all three labels from it. A False here means "don't
    trust the three fields above," not "the scorer failed to run."""
    raw_output: str
