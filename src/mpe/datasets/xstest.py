"""Loader for the secondary MVP benchmark: XSTest (EN native, KO draft).

English reads from a local snapshot of the canonical GitHub source (see
docs/phase1-notes.md section 3). Korean reads from a small, explicitly
unreviewed draft translation (see data/xstest_ko_draft/PROVENANCE.md) --
callers that need more items than exist in that draft get a clear error,
not a silently truncated sample.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mpe.datasets.base import DatasetLoader, deterministic_sample
from mpe.datasets.schema import BenchmarkItem, TaskType


class XSTestLoader(DatasetLoader):
    benchmark_name = "xstest"

    def __init__(
        self,
        en_csv_path: str | Path = "data/xstest/xstest_prompts_v2.csv",
        ko_jsonl_path: str | Path = "data/xstest_ko_draft/xstest_ko_draft.jsonl",
    ):
        self.en_csv_path = Path(en_csv_path)
        self.ko_jsonl_path = Path(ko_jsonl_path)

    def _load_en_rows(self) -> dict[str, dict]:
        with open(self.en_csv_path, newline="", encoding="utf-8") as f:
            return {row["id"]: row for row in csv.DictReader(f)}

    def _load_ko_rows(self) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        with open(self.ko_jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows[str(record["parallel_item_id"])] = record
        return rows

    def load(
        self, language: str, limit: int | None = None, seed: int = 0
    ) -> list[BenchmarkItem]:
        if language == "en":
            rows_by_id = self._load_en_rows()
        elif language == "ko":
            rows_by_id = self._load_ko_rows()
        else:
            raise ValueError(f"Unsupported language '{language}' for {self.benchmark_name}")

        chosen_ids = deterministic_sample(list(rows_by_id.keys()), limit, seed)

        items = []
        for item_id in chosen_ids:
            row = rows_by_id[item_id]
            items.append(
                BenchmarkItem(
                    item_id=f"{self.benchmark_name}:{language}:{item_id}",
                    benchmark=self.benchmark_name,
                    language=language,
                    parallel_item_id=item_id,
                    task_type=TaskType.REFUSAL_CLASSIFICATION,
                    prompt=row["prompt"],
                    expected_label=row["label"],
                    metadata={
                        "type": row.get("type"),
                        "focus": row.get("focus"),
                        "note": row.get("note"),
                        "translation_status": row.get(
                            "translation_status", "native_or_canonical"
                        ),
                    },
                )
            )
        return items
