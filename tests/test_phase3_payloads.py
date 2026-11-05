"""phase3/fixtures/payloads.json must match the Section 4.2 table in
docs/phase3-visual-injection.md character for character. The table there,
not this test, is the source of truth: this test parses the table out of
the doc itself rather than hardcoding a second transcription, so a future
edit to the doc's table (or to the fixture) that lets them drift apart is
caught here instead of silently shipping mismatched payloads."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "phase3-visual-injection.md"
PAYLOADS_PATH = REPO_ROOT / "phase3" / "fixtures" / "payloads.json"


def _extract_section_4_2_table(doc_text: str) -> dict[str, dict[str, str]]:
    start = doc_text.index("### 4.2 Extra line (payload)")
    end = doc_text.index("### 4.3 Channels", start)
    section = doc_text[start:end]

    table: dict[str, dict[str, str]] = {}
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        raw_id, en, ko = cells
        if raw_id == "ID":
            continue
        m = re.match(r"`([^`]+)`", raw_id)
        if not m:
            continue
        table[m.group(1)] = {"en": en, "ko": ko}
    return table


def test_payloads_json_matches_section_4_2_table_exactly():
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    expected = _extract_section_4_2_table(doc_text)

    assert set(expected.keys()) == {"neutral", "P1", "P2"}, (
        f"expected to find exactly neutral/P1/P2 rows in the Section 4.2 table, found {sorted(expected.keys())}"
    )

    actual = json.loads(PAYLOADS_PATH.read_text(encoding="utf-8"))
    assert actual == expected
