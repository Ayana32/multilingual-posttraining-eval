import csv

import pytest

from mpe.analysis.comprehension import (
    SHEET_COLUMNS,
    AnnotationError,
    export_annotation_sheet,
    summarize_annotations,
)
from mpe.checkpoints.schema import CheckpointStage
from mpe.storage.store import ResultStore
from tests.unit.test_bootstrap import _record


def _setup(tmp_path):
    store = ResultStore(results_dir=tmp_path / "results")
    store.write([
        _record("ko_sft", "k0", "0", True, language="ko", prompt="KO 0"),
        _record("ko_sft", "k1", "1", False, language="ko", prompt="KO 1"),
        _record("ko_sft", "k2", "2", False, language="ko", prompt="KO 2"),
        _record("ko_sft", "k3", "3", True, label="unharmful", language="ko"),
    ])
    store.write([_record("en_sft", f"e{i}", str(i), True, language="en", prompt=f"EN {i}") for i in range(4)])
    return store


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)


class TestExport:
    def test_exports_only_harmful_target_language_rows_without_scorer_columns(self, tmp_path):
        store = _setup(tmp_path)
        path = export_annotation_sheet(
            store, ["ko_sft"], tmp_path / "sheet.csv", reference_run_ids=["en_sft"]
        )
        rows = _read(path)
        assert [r["parallel_item_id"] for r in rows] == ["0", "1", "2"]
        assert not any(c.startswith("scorer") for c in rows[0])
        assert rows[1]["reference_prompt"] == "EN 1"

    def test_empty_run_raises(self, tmp_path):
        store = _setup(tmp_path)
        with pytest.raises(AnnotationError):
            export_annotation_sheet(store, ["missing"], tmp_path / "s.csv")


class TestSummarize:
    def test_agreement_breakdown_and_blank_rows(self, tmp_path):
        store = _setup(tmp_path)
        path = export_annotation_sheet(store, ["ko_sft"], tmp_path / "sheet.csv")
        rows = _read(path)
        # item 0: scorer refusal, human refusal (agree)
        rows[0].update(human_refusal="refusal", understood_request="yes", response_language="ko")
        # item 1: scorer non-refusal, human refusal (scorer missed a refusal)
        rows[1].update(human_refusal="Refusal", understood_request="yes", response_language="ko")
        # item 2: left blank (unannotated)
        _write(path, rows)

        s = summarize_annotations(store, path).runs[0]
        assert s.n_rows == 3 and s.n_annotated == 2
        assert s.human_refusal_counts == {"refusal": 2}
        assert s.n_agreement_pairs == 2
        assert s.agreement_rate == 0.5
        assert s.scorer_compliance_human_refusal == 1

    def test_non_refusal_understanding_breakdown(self, tmp_path):
        store = _setup(tmp_path)
        path = export_annotation_sheet(store, ["ko_sft"], tmp_path / "sheet.csv")
        rows = _read(path)
        rows[0].update(human_refusal="refusal")
        rows[1].update(human_refusal="compliance", understood_request="yes")
        rows[2].update(human_refusal="partial", understood_request="no")
        _write(path, rows)
        s = summarize_annotations(store, path).runs[0]
        assert s.understood_counts_among_non_refusals == {"no": 1, "yes": 1}
        # partial excluded from agreement
        assert s.n_agreement_pairs == 2
        assert s.agreement_rate == 1.0

    def test_invalid_label_raises_with_row_number(self, tmp_path):
        store = _setup(tmp_path)
        path = export_annotation_sheet(store, ["ko_sft"], tmp_path / "sheet.csv")
        rows = _read(path)
        rows[0].update(human_refusal="maybe")
        _write(path, rows)
        with pytest.raises(AnnotationError, match="row 2"):
            summarize_annotations(store, path)

    def test_row_not_in_store_raises(self, tmp_path):
        store = _setup(tmp_path)
        path = export_annotation_sheet(store, ["ko_sft"], tmp_path / "sheet.csv")
        rows = _read(path)
        rows[0]["parallel_item_id"] = "999"
        _write(path, rows)
        with pytest.raises(AnnotationError, match="999"):
            summarize_annotations(store, path)
