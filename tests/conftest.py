"""Shared fixtures: tiny, in-memory-built stand-ins for the real datasets.

Real production data files (data/xstest/*, data/xstest_ko_draft/*) are small
enough to just read directly in tests -- no fixture needed for those. For
PolyGuardPrompts (parquet, downloaded) and Belebele (paginated API, cached
as JSONL) we build tiny local files here instead, so unit/integration tests
never touch the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tiny_polyguard_parquet(tmp_path: Path) -> Path:
    rows = []
    for item_id in range(3):
        rows.append(
            {
                "id": item_id,
                "language": "English",
                "prompt": f"EN prompt {item_id}",
                "prompt_harm_label": "harmful" if item_id == 0 else "unharmful",
                "adversarial": item_id == 0,
                "subcategory": "test",
            }
        )
        rows.append(
            {
                "id": item_id,
                "language": "Korean",
                "prompt": f"KO 프롬프트 {item_id}",
                "prompt_harm_label": "harmful" if item_id == 0 else "unharmful",
                "adversarial": item_id == 0,
                "subcategory": "test",
            }
        )
    path = tmp_path / "polyguard_fixture.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture
def tiny_belebele_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "belebele_cache"
    cache_dir.mkdir()
    for config, question_text in [("eng_Latn", "What is the answer?"), ("kor_Hang", "정답은 무엇인가?")]:
        rows = []
        for i in range(3):
            rows.append(
                {
                    "link": f"https://example.org/passage{i}",
                    "question_number": 1,
                    "flores_passage": f"Passage {i} text.",
                    "question": question_text,
                    "mc_answer1": "A",
                    "mc_answer2": "B",
                    "mc_answer3": "C",
                    "mc_answer4": "D",
                    "correct_answer_num": "2",
                }
            )
        with open(cache_dir / f"{config}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return cache_dir
