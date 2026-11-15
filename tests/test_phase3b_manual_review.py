"""Phase 3b manual-review CSV (docs/phase3b-replication.md Section 10):

- the canonical order is 440 rows (280 arm A + 160 arm B) in the Section 10
  ordering;
- the 40 body-sampled rows reproduce the explicit list in Section 10 exactly
  (random.Random(20261112).sample(range(440), 40) over the canonical ordering);
- body_line_by_line is empty for exactly those 40 and "n/a" for the other 400;
  every other check cell is empty.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mr_rev = _load("phase3b_manual_review", "phase3b/make_manual_review.py")

# The 40 images listed verbatim in docs/phase3b-replication.md Section 10.
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


def test_canonical_order_is_440_arm_a_then_arm_b():
    entries = mr_rev.canonical_entries()
    assert len(entries) == 440
    arm_a = [e for e in entries if e["image_id"].startswith("A|")]
    arm_b = [e for e in entries if e["image_id"].startswith("B|")]
    assert len(arm_a) == 280 and len(arm_b) == 160
    # arm A comes entirely before arm B.
    assert all(e["image_id"].startswith("A|") for e in entries[:280])
    assert all(e["image_id"].startswith("B|") for e in entries[280:])
    # first arm A entry is clean/eval_01; first arm B is NotoSansKR/P1_en/eval_01.
    assert entries[0]["image_id"] == "A|clean|phase3b_eval_01"
    assert entries[280]["image_id"] == "B|NotoSansKR|P1_en|phase3b_eval_01"


def test_sampled_40_match_the_preregistered_list():
    entries = mr_rev.canonical_entries()
    sampled = {entries[i]["image_id"] for i in mr_rev.sampled_indices()}
    assert len(sampled) == 40
    assert sampled == PREREG_SAMPLE_40


def _variant_of(image_id: str) -> str:
    parts = image_id.split("|")
    return parts[1] if parts[0] == "A" else parts[2]


def test_csv_rows_header_and_body_sample(tmp_path):
    out = tmp_path / "g3_manual_review.csv"
    rows = mr_rev.write_csv(out)
    assert len(rows) == 440
    with open(out, newline="", encoding="utf-8") as f:
        disk = list(csv.DictReader(f))
    assert len(disk) == 440
    assert list(disk[0].keys()) == mr_rev.COLUMNS
    assert "footer_reference" in mr_rev.COLUMNS

    # body_line_by_line: empty for all 40 sampled rows (clean included -- clean
    # has a body, Section 10 (c) samples two clean images for it); n/a
    # otherwise. notes is always empty.
    n_body_empty = 0
    for r in disk:
        assert r["notes"] == ""
        if r["body_line_by_line"] == "":
            n_body_empty += 1
            assert r["image_id"] in PREREG_SAMPLE_40
    assert n_body_empty == 40
    # the two sampled clean rows are editable for the body check specifically.
    by_id = {r["image_id"]: r for r in disk}
    for iid in ("A|clean|phase3b_eval_09", "A|clean|phase3b_eval_13"):
        assert by_id[iid]["body_line_by_line"] == ""
        # every OTHER clean check column stays n/a
        for c in mr_rev.CHECK_COLUMNS:
            if c != "body_line_by_line":
                assert by_id[iid][c] == "n/a", (iid, c)


def test_na_prefills_by_variant(tmp_path):
    rows = mr_rev.write_csv(tmp_path / "x.csv")
    by_id = {r["image_id"]: r for r in rows}

    def row(iid):
        return by_id[iid]

    # clean: every check column n/a.
    for c in mr_rev.CHECK_COLUMNS:
        assert row("A|clean|phase3b_eval_01")[c] == "n/a", c

    # neutral_en: korean_renders, receipt_copy_one_line, address_one_line n/a;
    # footer checks remain empty (it has a footer).
    r = row("A|neutral_en|phase3b_eval_01")
    assert r["korean_renders"] == r["receipt_copy_one_line"] == r["address_one_line"] == "n/a"
    assert r["footer_not_cut_or_overlapping"] == "" and r["footer_line_by_line"] == ""

    # neutral_ko: only receipt_copy_one_line, address_one_line n/a; korean stays.
    r = row("A|neutral_ko|phase3b_eval_01")
    assert r["receipt_copy_one_line"] == r["address_one_line"] == "n/a"
    assert r["korean_renders"] == ""

    # P1_en/P2_en (arm A and arm B): korean_renders n/a, address/receipt stay.
    for iid in ("A|P1_en|phase3b_eval_01", "A|P2_en|phase3b_eval_01",
                "B|NotoSansKR|P1_en|phase3b_eval_01", "B|NanumGothic|P1_en|phase3b_eval_01"):
        r = row(iid)
        assert r["korean_renders"] == "n/a", iid
        assert r["receipt_copy_one_line"] == "" and r["address_one_line"] == "", iid

    # P1_ko/P2_ko: no payload-specific n/a (Korean + address + subject all apply).
    for iid in ("A|P1_ko|phase3b_eval_01", "A|P2_ko|phase3b_eval_01", "B|NotoSansKR|P1_ko|phase3b_eval_01"):
        r = row(iid)
        assert r["korean_renders"] == r["receipt_copy_one_line"] == r["address_one_line"] == ""


def test_footer_reference_column(tmp_path):
    rows = mr_rev.write_csv(tmp_path / "x.csv")
    by_id = {r["image_id"]: r for r in rows}
    # clean has no footer -> empty.
    assert by_id["A|clean|phase3b_eval_01"]["footer_reference"] == ""
    # arm A P1_en footer is 3 lines (2 " / " separators); arm B NotoSansKR P1_en is 4.
    assert by_id["A|P1_en|phase3b_eval_01"]["footer_reference"].count(" / ") == 2
    assert by_id["B|NotoSansKR|P1_en|phase3b_eval_01"]["footer_reference"].count(" / ") == 3
    # the reference actually contains the payload's target address.
    assert "accounts-archive@example.com" in by_id["A|P1_ko|phase3b_eval_01"]["footer_reference"]


def test_count_empty_per_column(tmp_path):
    rows = mr_rev.write_csv(tmp_path / "x.csv")
    counts = mr_rev.count_empty_per_column(rows)
    assert set(counts) == set(mr_rev.COLUMNS)
    # image_id/image_path/footer_reference(except clean) are filled; notes all empty.
    assert counts["image_id"] == 0 and counts["image_path"] == 0
    assert counts["notes"] == 440
    # footer_reference empty only for the 40 clean images.
    assert counts["footer_reference"] == 40
