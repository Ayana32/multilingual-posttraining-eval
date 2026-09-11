"""Loader for the primary MVP benchmark: PolyGuardPrompts (EN/KO).

See docs/phase1-notes.md section 1 for how the Korean subset size (1,725),
label balance, and the shared `id` parallel key were verified directly
against the downloaded parquet file rather than the dataset card.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from mpe.datasets.base import DatasetLoader, deterministic_sample
from mpe.datasets.schema import BenchmarkItem, TaskType

_LANGUAGE_NAME = {"en": "English", "ko": "Korean"}

_PARQUET_URL = (
    "https://huggingface.co/datasets/ToxicityPrompts/PolyGuardPrompts/"
    "resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)


class PolyGuardPromptsLoader(DatasetLoader):
    benchmark_name = "polyguard_prompts"

    def __init__(self, cache_path: str | Path = "data/cache/polyguard_prompts.parquet"):
        self.cache_path = Path(cache_path)

    def _ensure_cached(self) -> Path:
        if not self.cache_path.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(_PARQUET_URL, timeout=120)
            resp.raise_for_status()
            self.cache_path.write_bytes(resp.content)
        return self.cache_path

    def load(
        self,
        language: str,
        limit: int | None = None,
        seed: int = 0,
        item_ids: list[str] | None = None,
    ) -> list[BenchmarkItem]:
        if language not in _LANGUAGE_NAME:
            raise ValueError(
                f"Unsupported language '{language}' for {self.benchmark_name}; "
                f"supported: {list(_LANGUAGE_NAME)}"
            )
        df = pd.read_parquet(self._ensure_cached())
        subset = df[df["language"] == _LANGUAGE_NAME[language]]

        if item_ids is not None:
            # Frozen-sample path (e.g. Phase 2B's checked-in manifest): use
            # exactly these ids instead of re-deriving a sample via
            # deterministic_sample(). limit/seed are ignored when item_ids
            # is given -- the manifest is the authoritative sample. Matched
            # as strings since parallel_item_id is a string everywhere else
            # in this codebase; the raw `id` column's native dtype (int64)
            # is irrelevant here.
            available = set(subset["id"].astype(str))
            missing = sorted(set(item_ids) - available)
            if missing:
                raise ValueError(
                    f"item_ids requested but not present in {language} "
                    f"{self.benchmark_name}: {missing[:10]}"
                    f"{'...' if len(missing) > 10 else ''}"
                )
            chosen_ids_str = set(item_ids)
            subset = subset[subset["id"].astype(str).isin(chosen_ids_str)].sort_values("id")
        else:
            # `id` is confirmed shared across every language's rows for the same
            # underlying (translated) prompt -- sampling on it here, independent
            # of which language is being loaded, is what keeps EN/KO calls paired.
            chosen_ids = deterministic_sample(subset["id"].tolist(), limit, seed)
            subset = subset[subset["id"].isin(chosen_ids)].sort_values("id")

        items = []
        for _, row in subset.iterrows():
            # Missing prompt_harm_label rows (26 unresolved items -- see
            # docs/phase1-notes.md) must become expected_label=None. Whether
            # pandas/pyarrow surfaces a missing value as Python None or as
            # float('nan') is a pandas-version implementation detail, not
            # something this code should rely on (it changed across the
            # pandas 2->3 boundary) -- pd.isna() catches both uniformly.
            raw_label = row["prompt_harm_label"]
            expected_label = None if pd.isna(raw_label) else raw_label
            items.append(
                BenchmarkItem(
                    item_id=f"{self.benchmark_name}:{language}:{row['id']}",
                    benchmark=self.benchmark_name,
                    language=language,
                    parallel_item_id=str(row["id"]),
                    task_type=TaskType.REFUSAL_CLASSIFICATION,
                    prompt=row["prompt"],
                    expected_label=expected_label,
                    metadata={
                        "adversarial": bool(row["adversarial"]),
                        "subcategory": row.get("subcategory"),
                    },
                )
            )
        return items
