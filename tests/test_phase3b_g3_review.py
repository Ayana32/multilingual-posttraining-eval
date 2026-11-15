"""Phase 3b G3 (b)+(c) review viewer (phase3b/g3_review.py).

Covers the pure review logic (n/a locking, crop boxes from the committed draw
bboxes, the remaining-cells counter, and the --export merge with its refusals)
and a light localhost HTTP smoke test. No evaluation runs; reads only committed
artifacts (report/phase3b/g3a_results.json, report/phase3b/g3_manual_review.csv,
phase3b/data/images).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rv = _load("phase3b_g3_review", "phase3b/g3_review.py")

PREREG_SAMPLE_40 = {
    "A|clean|phase3b_eval_09", "A|clean|phase3b_eval_13",
    "A|neutral_en|phase3b_eval_07", "A|neutral_en|phase3b_eval_16",
    "A|neutral_en|phase3b_eval_20", "A|neutral_en|phase3b_eval_22",
    "A|neutral_ko|phase3b_eval_11", "A|neutral_ko|phase3b_eval_26",
    "A|neutral_ko|phase3b_eval_27", "A|neutral_ko|phase3b_eval_28",
    "A|neutral_ko|phase3b_eval_39", "A|P1_en|phase3b_eval_05",
    "A|P1_en|phase3b_eval_08", "A|P1_en|phase3b_eval_36",
    "A|P1_ko|phase3b_eval_08", "A|P1_ko|phase3b_eval_15",
    "A|P1_ko|phase3b_eval_22", "A|P1_ko|phase3b_eval_36",
    "A|P2_en|phase3b_eval_21", "A|P2_en|phase3b_eval_23",
    "A|P2_en|phase3b_eval_24", "A|P2_ko|phase3b_eval_03",
    "A|P2_ko|phase3b_eval_13", "A|P2_ko|phase3b_eval_19",
    "A|P2_ko|phase3b_eval_27", "A|P2_ko|phase3b_eval_31",
    "A|P2_ko|phase3b_eval_37", "A|P2_ko|phase3b_eval_38",
    "B|NotoSansKR|P1_en|phase3b_eval_30", "B|NotoSansKR|P1_en|phase3b_eval_39",
    "B|NotoSansKR|P1_ko|phase3b_eval_24", "B|NotoSansKR|P1_ko|phase3b_eval_33",
    "B|NanumGothic|P1_en|phase3b_eval_14", "B|NanumGothic|P1_en|phase3b_eval_16",
    "B|NanumGothic|P1_en|phase3b_eval_30", "B|NanumGothic|P1_ko|phase3b_eval_05",
    "B|NanumGothic|P1_ko|phase3b_eval_07", "B|NanumGothic|P1_ko|phase3b_eval_20",
    "B|NanumGothic|P1_ko|phase3b_eval_30", "B|NanumGothic|P1_ko|phase3b_eval_35",
}


@pytest.fixture(scope="module")
def rows():
    return rv.load_rows()


@pytest.fixture(scope="module")
def g3a():
    return rv.load_g3a()


# --------------------------------------------------------------------------- #
# parsing / cells
# --------------------------------------------------------------------------- #
def test_parse_and_cell():
    assert rv.parse_image_id("A|P1_ko|phase3b_eval_05") == {
        "arm": "A", "font": None, "variant": "P1_ko", "receipt_id": "phase3b_eval_05"}
    assert rv.parse_image_id("B|NanumGothic|P1_en|phase3b_eval_14") == {
        "arm": "B", "font": "NanumGothic", "variant": "P1_en", "receipt_id": "phase3b_eval_14"}
    assert rv.cell_of("A|P1_ko|phase3b_eval_05") == "A|P1_ko"
    assert rv.cell_of("B|NanumGothic|P1_en|phase3b_eval_14") == "B|NanumGothic|P1_en"


def test_cell_order_11_cells(rows):
    cells = rv.cell_order(rows)
    assert len(cells) == 11
    assert cells[0] == "A|clean"
    assert cells[:7] == ["A|clean", "A|neutral_en", "A|neutral_ko",
                         "A|P1_en", "A|P1_ko", "A|P2_en", "A|P2_ko"]
    assert set(cells[7:]) == {"B|NotoSansKR|P1_en", "B|NotoSansKR|P1_ko",
                              "B|NanumGothic|P1_en", "B|NanumGothic|P1_ko"}


# --------------------------------------------------------------------------- #
# editable (non-n/a) checks per variant -- the locking rule
# --------------------------------------------------------------------------- #
def test_editable_checks_match_na_rules(rows):
    by_id = {r["image_id"]: r for r in rows}

    def edit(iid):
        return set(rv.editable_checks(by_id[iid], rv.CELL_CHECKS))

    # clean: nothing editable in the cell view (all n/a).
    assert edit("A|clean|phase3b_eval_01") == set()
    # neutral_en: korean/receipt/address n/a -> only the three footer-shape checks.
    assert edit("A|neutral_en|phase3b_eval_01") == {
        "footer_not_cut_or_overlapping", "body_footer_spacing_ok", "footer_line_by_line"}
    # neutral_ko: korean stays; receipt/address n/a.
    assert edit("A|neutral_ko|phase3b_eval_01") == {
        "korean_renders", "footer_not_cut_or_overlapping",
        "body_footer_spacing_ok", "footer_line_by_line"}
    # P1_en: korean n/a; the rest editable.
    assert edit("A|P1_en|phase3b_eval_01") == {
        "receipt_copy_one_line", "address_one_line", "footer_not_cut_or_overlapping",
        "body_footer_spacing_ok", "footer_line_by_line"}
    # P1_ko: all six editable.
    assert edit("A|P1_ko|phase3b_eval_01") == set(rv.CELL_CHECKS)
    assert edit("B|NanumGothic|P1_en|phase3b_eval_01") == {
        "receipt_copy_one_line", "address_one_line", "footer_not_cut_or_overlapping",
        "body_footer_spacing_ok", "footer_line_by_line"}


# --------------------------------------------------------------------------- #
# record_judgment
# --------------------------------------------------------------------------- #
def test_record_judgment_validation():
    p = {"judgments": {}}
    rv.record_judgment(p, "A|P1_ko|phase3b_eval_01", "korean_renders", "pass", "")
    assert p["judgments"]["A|P1_ko|phase3b_eval_01"]["korean_renders"]["value"] == "pass"
    # fail requires a note
    with pytest.raises(ValueError):
        rv.record_judgment(p, "A|P1_ko|phase3b_eval_01", "korean_renders", "fail", "  ")
    rv.record_judgment(p, "A|P1_ko|phase3b_eval_01", "korean_renders", "fail", "tofu glyph")
    assert p["judgments"]["A|P1_ko|phase3b_eval_01"]["korean_renders"]["note"] == "tofu glyph"
    # bad value / unknown column
    with pytest.raises(ValueError):
        rv.record_judgment(p, "x", "korean_renders", "n/a", "")
    with pytest.raises(ValueError):
        rv.record_judgment(p, "x", "not_a_column", "pass", "")


# --------------------------------------------------------------------------- #
# crop boxes from the committed draw bboxes
# --------------------------------------------------------------------------- #
def test_crop_box_footer_and_boundary(g3a):
    # clean has no footer -> no footer or boundary crop; body crop exists.
    clean = g3a["A|clean|phase3b_eval_01"]
    assert rv.crop_box(clean, "footer") is None
    assert rv.crop_box(clean, "boundary") is None
    assert rv.crop_box(clean, "body") is not None

    e = g3a["B|NotoSansKR|P1_en|phase3b_eval_01"]
    footer = rv.crop_box(e, "footer")
    boundary = rv.crop_box(e, "boundary")
    body = rv.crop_box(e, "body")
    for b in (footer, boundary, body):
        assert b is not None
        assert 0 <= b[0] < b[2] <= e["width"]
        assert 0 <= b[1] < b[3] <= e["height"]
    # boundary starts at/above the footer (it includes the Total line) and
    # reaches the footer's bottom.
    assert boundary[1] <= footer[1]
    assert boundary[3] == footer[3]
    # body is above the footer.
    assert body[3] <= footer[3]


def test_crop_box_margin_and_clamp(g3a):
    e = g3a["A|P1_ko|phase3b_eval_01"]
    footer_draws = [d["bbox"] for d in e["draws"] if d["kind"] == "footer"]
    raw = rv._union(footer_draws)
    box = rv.crop_box(e, "footer", margin=14)
    assert box[0] == max(0, raw[0] - 14)
    assert box[2] == min(e["width"], raw[2] + 14)
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= e["width"] and box[3] <= e["height"]


def test_render_crop_png_is_png(g3a):
    e = g3a["A|P1_ko|phase3b_eval_01"]
    png = rv.render_crop_png(rv._image_path_for("A|P1_ko|phase3b_eval_01"),
                             rv.crop_box(e, "footer"))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# sampled set + remaining counter
# --------------------------------------------------------------------------- #
def test_sampled_image_ids(rows):
    assert set(rv.sampled_image_ids(rows)) == PREREG_SAMPLE_40
    assert len(rv.sampled_image_ids(rows)) == 40


def _complete_progress(rows):
    """A progress object that judges every editable cell-view check (pass) and
    every editable body check (pass)."""
    p = {"judgments": {}}
    for r in rows:
        for c in rv.editable_checks(r, rv.CELL_CHECKS):
            rv.record_judgment(p, r["image_id"], c, "pass", "")
    for iid in rv.sampled_image_ids(rows):
        row = next(x for x in rows if x["image_id"] == iid)
        if row.get(rv.BODY_CHECK) != "n/a":
            rv.record_judgment(p, iid, rv.BODY_CHECK, "pass", "")
    return p


def test_remaining_cells_counter(rows):
    empty = {"judgments": {}}
    # clean cell is all n/a -> already done; the other 10 cells remain.
    assert rv.remaining_cells(rows, empty) == 10
    full = _complete_progress(rows)
    assert rv.remaining_cells(rows, full) == 0
    assert rv.remaining_body(rows, full) == 0
    # body: 38 editable (the 2 sampled clean are n/a).
    assert rv.remaining_body(rows, empty) == 38


# --------------------------------------------------------------------------- #
# --export : success + refusals
# --------------------------------------------------------------------------- #
def test_export_success(rows, tmp_path):
    prog = tmp_path / "prog.json"
    rv.save_progress(_complete_progress(rows), prog)
    out = tmp_path / "out.csv"
    summary = rv.export(rv.CSV_PATH, prog, out)
    # structure preserved
    with open(out, newline="", encoding="utf-8") as f:
        disk = list(csv.DictReader(f))
    assert len(disk) == 440
    assert [r["image_id"] for r in disk] == rv._canonical_ids()
    assert list(disk[0].keys()) == rv.COLUMNS
    # no empty check cells remain; only pass / n/a (all judged pass here)
    for r in disk:
        for col in rv.ALL_CHECKS:
            assert r[col] in ("pass", "n/a")
    # value counts: korean_renders n/a count == 240 (the Section-10 prefill:
    # clean + *_en footers + arm B *_en x 2 fonts; the other 200 become pass).
    assert summary["value_counts"]["korean_renders"]["n/a"] == 240
    assert summary["value_counts"]["korean_renders"]["pass"] == 200
    assert summary["fails"] == []


def test_export_reports_fails_with_notes(rows, tmp_path):
    p = _complete_progress(rows)
    # flip one real check to fail with a note
    rv.record_judgment(p, "A|P1_ko|phase3b_eval_02", "footer_line_by_line",
                       "fail", "second footer line clipped")
    prog = tmp_path / "prog.json"
    rv.save_progress(p, prog)
    out = tmp_path / "out.csv"
    summary = rv.export(rv.CSV_PATH, prog, out)
    assert ("A|P1_ko|phase3b_eval_02", "footer_line_by_line",
            "second footer line clipped") in summary["fails"]
    assert summary["value_counts"]["footer_line_by_line"]["fail"] == 1
    # the note lands in the row's notes column
    by_id = {r["image_id"]: r for r in summary["rows"]}
    assert "footer_line_by_line: second footer line clipped" in by_id["A|P1_ko|phase3b_eval_02"]["notes"]


def test_export_refuses_empty_required_cell(rows, tmp_path):
    p = _complete_progress(rows)
    # drop one judgment -> an empty required cell
    del p["judgments"]["A|P1_ko|phase3b_eval_01"]["korean_renders"]
    prog = tmp_path / "prog.json"
    rv.save_progress(p, prog)
    with pytest.raises(rv.ExportError, match="empty required cell"):
        rv.export(rv.CSV_PATH, prog, tmp_path / "out.csv")


def test_export_refuses_row_count_change(rows, tmp_path):
    short = tmp_path / "short.csv"
    rv.write_rows(rows[:-1], short)
    prog = tmp_path / "prog.json"
    rv.save_progress(_complete_progress(rows), prog)
    with pytest.raises(rv.ExportError, match="row count changed"):
        rv.export(short, prog, tmp_path / "out.csv")


def test_export_refuses_id_or_order_change(rows, tmp_path):
    reordered = list(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    scrambled = tmp_path / "scrambled.csv"
    rv.write_rows(reordered, scrambled)
    prog = tmp_path / "prog.json"
    rv.save_progress(_complete_progress(rows), prog)
    with pytest.raises(rv.ExportError, match="id/order changed"):
        rv.export(scrambled, prog, tmp_path / "out.csv")


def test_export_refuses_bad_value(rows, tmp_path):
    p = _complete_progress(rows)
    # inject an illegal value directly into progress (bypassing record_judgment)
    p["judgments"]["A|P1_ko|phase3b_eval_01"]["korean_renders"] = {"value": "maybe", "note": ""}
    prog = tmp_path / "prog.json"
    rv.save_progress(p, prog)
    with pytest.raises(rv.ExportError, match="bad value"):
        rv.export(rv.CSV_PATH, prog, tmp_path / "out.csv")


def test_export_refuses_fail_without_note(rows, tmp_path):
    p = _complete_progress(rows)
    p["judgments"]["A|P1_ko|phase3b_eval_01"]["korean_renders"] = {"value": "fail", "note": ""}
    prog = tmp_path / "prog.json"
    rv.save_progress(p, prog)
    with pytest.raises(rv.ExportError, match="fail without note"):
        rv.export(rv.CSV_PATH, prog, tmp_path / "out.csv")


# --------------------------------------------------------------------------- #
# localhost HTTP smoke test
# --------------------------------------------------------------------------- #
def test_http_smoke(rows, g3a, tmp_path):
    rv.ReviewHandler.rows = rows
    rv.ReviewHandler.g3a = g3a
    rv.ReviewHandler.progress = {"judgments": {}}
    rv.ReviewHandler.progress_path = tmp_path / "prog.json"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), rv.ReviewHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert b"Cells remaining" in urllib.request.urlopen(base + "/").read()
        assert b"A|P1_ko" in urllib.request.urlopen(base + "/cell?c=A|P1_ko").read()
        assert b"Body line-by-line" in urllib.request.urlopen(base + "/body").read()
        png = urllib.request.urlopen(base + "/crop?id=A|P1_ko|phase3b_eval_01&kind=footer").read()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        # POST a judgment; it persists and the counter drops
        body = json.dumps({"image_id": "A|P1_ko|phase3b_eval_01",
                           "column": "korean_renders", "value": "pass", "note": ""}).encode()
        req = urllib.request.Request(base + "/judge", data=body,
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req).read())
        assert resp["ok"] is True
        assert (tmp_path / "prog.json").exists()
        # a fail without a note is rejected by the server (nothing stored)
        bad = json.dumps({"image_id": "A|P1_ko|phase3b_eval_01",
                         "column": "address_one_line", "value": "fail", "note": ""}).encode()
        req2 = urllib.request.Request(base + "/judge", data=bad,
                                      headers={"Content-Type": "application/json"})
        resp2 = json.loads(urllib.request.urlopen(req2).read())
        assert resp2["ok"] is False
    finally:
        httpd.shutdown()
