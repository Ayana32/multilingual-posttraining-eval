"""Loader for the language-competence control: Belebele Korean (kor_Hang).

Not a safety benchmark -- this is the Tier-1 control described in
docs/phase0-findings.md section 5, used to gate whether an observed
cross-lingual behavioural difference reflects post-training or just weaker
Korean reading competence at that checkpoint.

The parallel key across languages is (link, question_number), NOT row index.
This was verified directly (docs/phase1-notes.md section 5's sibling check):
Belebele's per-language configs are not row-aligned -- row 0 of eng_Latn and
row 0 of kor_Hang are different passages entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from mpe.datasets.base import DatasetLoader, deterministic_sample
from mpe.datasets.schema import BenchmarkItem, TaskType

_CONFIG = {"en": "eng_Latn", "ko": "kor_Hang"}
_PAGE_SIZE = 100


class BelebeleKoLoader(DatasetLoader):
    benchmark_name = "belebele_ko_control"

    def __init__(self, cache_dir: str | Path = "data/cache/belebele"):
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, language: str) -> Path:
        return self.cache_dir / f"{_CONFIG[language]}.jsonl"

    def _ensure_cached(self, language: str) -> Path:
        path = self._cache_path(language)
        if path.exists():
            return path
        config = _CONFIG[language]
        rows: list[dict] = []
        offset = 0
        while True:
            url = (
                "https://datasets-server.huggingface.co/rows"
                f"?dataset=facebook/belebele&config={config}&split=test"
                f"&offset={offset}&length={_PAGE_SIZE}"
            )
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            page = resp.json()["rows"]
            if not page:
                break
            rows.extend(r["row"] for r in page)
            offset += _PAGE_SIZE
            if len(page) < _PAGE_SIZE:
                break
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _read_cached(self, language: str) -> dict[str, dict]:
        path = self._ensure_cached(language)
        rows: dict[str, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = f"{row['link']}#{row['question_number']}"
                rows[key] = row
        return rows

    def load(
        self, language: str, limit: int | None = None, seed: int = 0
    ) -> list[BenchmarkItem]:
        if language not in _CONFIG:
            raise ValueError(f"Unsupported language '{language}' for {self.benchmark_name}")
        rows_by_key = self._read_cached(language)
        chosen_keys = deterministic_sample(list(rows_by_key.keys()), limit, seed)

        items = []
        for key in chosen_keys:
            row = rows_by_key[key]
            choices = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
            items.append(
                BenchmarkItem(
                    item_id=f"{self.benchmark_name}:{language}:{key}",
                    benchmark=self.benchmark_name,
                    language=language,
                    parallel_item_id=key,
                    task_type=TaskType.MULTIPLE_CHOICE,
                    prompt=self._format_prompt(row),
                    choices=choices,
                    correct_choice_index=int(row["correct_answer_num"]) - 1,
                    metadata={"passage": row["flores_passage"]},
                )
            )
        return items

    @staticmethod
    def _format_prompt(row: dict) -> str:
        choices = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
        labeled = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices))
        return f"{row['flores_passage']}\n\nQuestion: {row['question']}\n{labeled}"
