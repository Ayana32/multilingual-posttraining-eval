"""Checks on Phase 3b's receipt generator (phase3b/make_receipts.py),
mirroring tests/test_phase3_receipts.py:

1. The pre-registered receipt-to-vendor assignment (docs/phase3b-replication.md
   Section 3) is reproduced exactly: generate_receipts()'s receipt_id -> vendor
   equals the 40-row table, in order. This is Section 3's "Planned test".
2. The new vendor pool has 40 distinct names and ZERO overlap with Phase 3's
   VENDORS, checked against the imported phase3.make_receipts.VENDORS list, not
   a hand-typed copy (Section 3).
3. The fixed seed reproduces identical receipts.jsonl and transcript hashes on
   a fresh rebuild, always.
4. The fixed seed reproduces identical image hashes too, but only when the
   current Pillow/FreeType versions match (PNG bytes can differ across
   Pillow/FreeType/OS versions; same caveat as tests/test_phase3_receipts.py).
5. Every rendered variant is receipt_set == "phase3b_eval" and arm "A".

Requires the "phase3" optional dependency group (Pillow); CI installs it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# phase3/ and phase3b/ both hold a module named "make_receipts". Load Phase 3's
# normally so it is cached under the name "make_receipts" (phase3b's own
# `import make_receipts as mr` then resolves to it), and load Phase 3b's under a
# distinct module name via its file path, so the two never collide in the
# import cache regardless of test order.
sys.path.insert(0, str(REPO_ROOT / "phase3"))
import make_receipts as mr  # noqa: E402  (Phase 3's generator)

_spec = importlib.util.spec_from_file_location(
    "phase3b_make_receipts", REPO_ROOT / "phase3b" / "make_receipts.py"
)
p3b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3b)


# The pre-registered assignment table, docs/phase3b-replication.md Section 3
# (lines "phase3b_eval_01: ..." through "phase3b_eval_40: ..."), transcribed.
EXPECTED_ASSIGNMENT = {
    "phase3b_eval_01": "Oakhaven Grocers",
    "phase3b_eval_02": "Hartwell Hardware",
    "phase3b_eval_03": "Shadowfen Bakery",
    "phase3b_eval_04": "Longmeadow Hardware",
    "phase3b_eval_05": "Ironbridge Diner",
    "phase3b_eval_06": "Hazelwood Cafe",
    "phase3b_eval_07": "Wrenfield Cafe",
    "phase3b_eval_08": "Clearwater Bakery",
    "phase3b_eval_09": "Windham Diner",
    "phase3b_eval_10": "Frostwick Pharmacy",
    "phase3b_eval_11": "Mossgate Stationery",
    "phase3b_eval_12": "Pebblebrook Books",
    "phase3b_eval_13": "Elmsworth Grocers",
    "phase3b_eval_14": "Blackfern Cafe",
    "phase3b_eval_15": "Brambleton Stationery",
    "phase3b_eval_16": "Driftwood Stationery",
    "phase3b_eval_17": "Thistledown Hardware",
    "phase3b_eval_18": "Ironwood Grocers",
    "phase3b_eval_19": "Ashford Electronics",
    "phase3b_eval_20": "Moonvale Pharmacy",
    "phase3b_eval_21": "Highmoor Pharmacy",
    "phase3b_eval_22": "Marrowgate Grocers",
    "phase3b_eval_23": "Greystone Cafe",
    "phase3b_eval_24": "Redwood Hardware",
    "phase3b_eval_25": "Timberline Diner",
    "phase3b_eval_26": "Rosemont Diner",
    "phase3b_eval_27": "Goldcrest Bakery",
    "phase3b_eval_28": "Dunmore Books",
    "phase3b_eval_29": "Thornfield Books",
    "phase3b_eval_30": "Nightshade Stationery",
    "phase3b_eval_31": "Whitfield Cafe",
    "phase3b_eval_32": "Ravenscroft Electronics",
    "phase3b_eval_33": "Fairview Electronics",
    "phase3b_eval_34": "Sunridge Bakery",
    "phase3b_eval_35": "Mapleridge Electronics",
    "phase3b_eval_36": "Sparrowdale Grocers",
    "phase3b_eval_37": "Sablewood Pharmacy",
    "phase3b_eval_38": "Larksfield Stationery",
    "phase3b_eval_39": "Stonebridge Hardware",
    "phase3b_eval_40": "Silverpine Books",
}


def test_assignment_reproduces_preregistered_table():
    receipts = p3b.generate_receipts()
    assert len(receipts) == 40
    got = {r["receipt_id"]: r["vendor"] for r in receipts}
    assert got == EXPECTED_ASSIGNMENT
    # Order, too: receipt ids in order, each vendor used exactly once.
    ids = [r["receipt_id"] for r in receipts]
    assert ids == [f"phase3b_eval_{i:02d}" for i in range(1, 41)]
    assert len({r["vendor"] for r in receipts}) == 40


def test_new_pool_is_distinct_and_disjoint_from_phase3_vendors():
    pool = p3b.NEW_VENDOR_POOL
    assert len(pool) == 40
    assert len(set(pool)) == 40
    # Checked against the imported Phase 3 VENDORS list, not a typed copy.
    assert set(pool) & set(mr.VENDORS) == set()
    assert len(mr.VENDORS) == 23


def test_every_row_is_phase3b_eval_set_and_arm_a(tmp_path):
    out_dir = tmp_path / "data"
    p3b.build(out_dir=out_dir)
    manifest = [json.loads(l) for l in (out_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    assert len(manifest) == 40 * 7  # 40 receipts x 7 variants
    assert all(r["receipt_set"] == "phase3b_eval" for r in manifest)
    assert all(r["arm"] == "A" and r["font"] is None for r in manifest)


@pytest.fixture(scope="module")
def two_builds(tmp_path_factory):
    a = tmp_path_factory.mktemp("p3b_a")
    b = tmp_path_factory.mktemp("p3b_b")
    p3b.build(out_dir=a)
    p3b.build(out_dir=b)
    load = lambda d: {
        (r["receipt_id"], r["variant"]): r
        for r in (json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip())
    }
    return {"a": a, "b": b, "ma": load(a), "mb": load(b)}


def test_same_seed_reproduces_receipts_and_transcript_hashes(two_builds):
    ma, mb = two_builds["ma"], two_builds["mb"]
    assert set(ma) == set(mb)
    for key in ma:
        assert ma[key]["transcript_sha256"] == mb[key]["transcript_sha256"], key
    ra = (two_builds["a"] / "receipts.jsonl").read_text(encoding="utf-8")
    rb = (two_builds["b"] / "receipts.jsonl").read_text(encoding="utf-8")
    assert ra == rb


def test_same_seed_reproduces_image_hashes_in_matching_environment(two_builds):
    ma, mb = two_builds["ma"], two_builds["mb"]
    env = json.loads((two_builds["a"] / "environment.json").read_text(encoding="utf-8"))
    current = mr.environment_info()
    if not (env["pillow_version"] == current["pillow_version"]
            and env["freetype_version"] == current["freetype_version"]):
        pytest.skip("Pillow/FreeType mismatch; transcript hashes are checked unconditionally.")
    for key in ma:
        assert ma[key]["image_sha256"] == mb[key]["image_sha256"], key
