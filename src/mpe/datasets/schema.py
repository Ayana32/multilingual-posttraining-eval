"""Typed schema for benchmark items.

One schema covers both task shapes used in Phase 1: free-generation items
scored by refusal/compliance (PolyGuardPrompts, XSTest) and multiple-choice
items scored by exact match (the Belebele-KO competence control). Keeping a
single BenchmarkItem type (rather than one class per benchmark) is what lets
the Evaluator, output parser, and metrics stay benchmark-agnostic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    REFUSAL_CLASSIFICATION = "refusal_classification"
    MULTIPLE_CHOICE = "multiple_choice"


class BenchmarkItem(BaseModel):
    item_id: str
    benchmark: str
    language: str
    parallel_item_id: str
    """Key shared across languages for the *same* underlying item.

    For PolyGuardPrompts this is the dataset's own integer `id` (confirmed
    shared across all 17 language rows for the same prompt). For XSTest it's
    the original English item id. For Belebele it's `link#question_number`
    (confirmed via direct verification that Belebele's per-language configs
    are NOT row-index aligned -- see docs/phase1-notes.md section 5's
    sibling check).
    """
    task_type: TaskType
    prompt: str
    choices: list[str] | None = None
    correct_choice_index: int | None = None
    expected_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
