from __future__ import annotations

from pydantic import BaseModel


class ParsedOutput(BaseModel):
    item_id: str
    is_parseable: bool
    language_match: bool | None = None
    refusal_label: str | None = None  # "refusal" | "compliance"
    mc_choice_index: int | None = None
    raw_completion: str
