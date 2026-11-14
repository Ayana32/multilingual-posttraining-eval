"""Phase 3b G0 additional comparison (docs/phase3b-replication.md Section 10):
the ordered mismatch rule. No model is run here; synthetic re-run directories
built from the committed records exercise the logic.

- It identifies exactly the 20 committed P1_ko image records.
- An identical re-run (same versions, same outputs) -> verdict "match", 20/20,
  and the re-run copy count equals Phase 3's 16.
- A version difference is the cause to fix first, even with identical outputs.
- Identical versions but a changed output -> "nondeterminism_finding", with the
  mismatch counted and the copy count reported.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import score as sc  # noqa: E402
import exploratory_m3 as em  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


g0 = _load("phase3b_g0_compare", "phase3b/g0_compare.py")


def _write_rerun(dirpath: Path, records: dict, env_versions: dict):
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / "records.jsonl", "w", encoding="utf-8") as f:
        for rid in sorted(records):
            f.write(json.dumps(records[rid], ensure_ascii=False, sort_keys=True) + "\n")
    (dirpath / "environment.json").write_text(json.dumps(env_versions, indent=2, sort_keys=True), encoding="utf-8")


def test_identifies_20_committed_p1ko_image():
    committed = g0.load_committed_p1ko_image()
    assert len(committed) == 20
    assert all(r["variant"] == "P1_ko" and r["channel"] == "image" for r in committed.values())


def test_identical_rerun_all_match_and_copy_count_16(tmp_path):
    committed = g0.load_committed_p1ko_image()
    env = g0.committed_env_versions()
    rerun_dir = tmp_path / "g0_rerun_check"
    _write_rerun(rerun_dir, committed, env)  # byte-identical re-run

    summary = g0.build_summary(rerun_dir)
    assert summary["verdict"] == "match"
    assert summary["n_match"] == 20 and summary["n_mismatch"] == 0
    assert summary["environment"]["match"] is True
    # cross-check: copy over these committed P1_ko image records is Phase 3's 16.
    assert summary["rerun_copy_count"] == 16
    assert summary["phase3_copy_count"] == 16


def test_version_mismatch_is_cause_first(tmp_path):
    committed = g0.load_committed_p1ko_image()
    env = dict(g0.committed_env_versions())
    env["torch_version"] = "9.9.9+cuXXX"  # a different serving stack
    rerun_dir = tmp_path / "g0_rerun_check"
    _write_rerun(rerun_dir, committed, env)  # outputs identical, versions differ

    summary = g0.build_summary(rerun_dir)
    assert summary["verdict"] == "version_mismatch_fix_first"
    assert summary["environment"]["match"] is False
    assert "torch_version" in summary["environment"]["diffs"]
    assert summary["rerun_copy_count"] is None  # not read until versions align


def test_output_diff_with_matching_versions_is_nondeterminism(tmp_path):
    committed = g0.load_committed_p1ko_image()
    env = g0.committed_env_versions()
    rerun = {rid: dict(rec) for rid, rec in committed.items()}
    # Flip one record's final_text so outputs differ on one receipt.
    victim = sorted(rerun)[0]
    rerun[victim] = dict(rerun[victim])
    rerun[victim]["final_text"] = rerun[victim]["final_text"] + " DRIFT"
    rerun_dir = tmp_path / "g0_rerun_check"
    _write_rerun(rerun_dir, rerun, env)

    summary = g0.build_summary(rerun_dir)
    assert summary["verdict"] == "nondeterminism_finding"
    assert summary["n_mismatch"] == 1 and summary["n_match"] == 19
    assert summary["rerun_copy_count"] is not None
    bad = [r for r in summary["per_receipt"] if r["receipt_id"] == victim][0]
    assert "final_text" in bad["differing_fields"]


def test_write_summary_emits_json_and_md(tmp_path):
    committed = g0.load_committed_p1ko_image()
    env = g0.committed_env_versions()
    rerun_dir = tmp_path / "g0_rerun_check"
    _write_rerun(rerun_dir, committed, env)
    summary = g0.build_summary(rerun_dir)
    jp, mp = g0.write_summary(summary, tmp_path / "out")
    assert jp.exists() and mp.exists()
    assert "Verdict" in mp.read_text(encoding="utf-8")
