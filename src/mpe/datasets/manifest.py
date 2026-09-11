"""Reads a frozen parallel-item-id manifest (see configs/samples/) -- the
mechanism Phase 2B uses to pin an exact, auditable sample instead of
re-deriving one from a seed on every run. A manifest is produced once
(filter-then-sample over a dataset loader's real data, e.g. excluding
null-label ids) and checked in; ExperimentRunner reads it via
load_id_manifest() and passes the resulting ids to a loader that supports
an `item_ids` override (currently PolyGuardPromptsLoader only).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_id_manifest(path: str | Path) -> list[str]:
    """Reads a manifest JSON file's `parallel_item_ids` list.

    Raises clearly (not silently) on a missing/malformed/duplicate-
    containing manifest -- a frozen sample is a research-integrity
    dependency, not an optional nicety, so a broken manifest must fail
    fast rather than silently produce a smaller or different sample.
    """
    raw = Path(path).read_text()
    data = json.loads(raw)
    ids = data.get("parallel_item_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Manifest at {path} has no non-empty 'parallel_item_ids' list")
    ids_str = [str(i) for i in ids]
    if len(ids_str) != len(set(ids_str)):
        raise ValueError(f"Manifest at {path} contains duplicate parallel_item_ids")
    return ids_str
