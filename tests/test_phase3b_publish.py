"""Phase 3b publish step (phase3b/publish_eval.py): it copies exactly one
successful directory per chunk into report/phase3b_eval/ and refuses on a
missing, short, over-long or duplicate chunk; the SHA256SUMS it writes verifies.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import exploratory_m3 as em  # noqa: E402  (verify_sha256sums)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pub = _load("phase3b_publish_eval", "phase3b/publish_eval.py")

PLAN = [{"name": "c1", "kind": "agent", "expected_count": 2}]


def _write_chunk(results_dir: Path, name: str, n_records: int, kind: str = "agent"):
    d = results_dir / name
    d.mkdir(parents=True, exist_ok=True)
    rf = pub.RECORD_FILE[kind]
    with open(d / rf, "w", encoding="utf-8") as f:
        for i in range(n_records):
            f.write(json.dumps({"i": i}) + "\n")
    (d / "environment.json").write_text(json.dumps({"git_dirty": False}) + "\n", encoding="utf-8")
    return d


def test_publish_success_copies_and_writes_verifiable_sha(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _write_chunk(results, "c1", 2)
    out = tmp_path / "report" / "phase3b_eval"
    copied = pub.publish(results, out, {"c1": 0}, PLAN)
    assert (out / "c1" / "records.jsonl").exists()
    assert (out / "c1" / "environment.json").exists()
    assert (out / "SHA256SUMS").exists()
    assert len(copied) == 2
    em.verify_sha256sums(out)  # must not raise


def test_publish_refuses_short_chunk(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _write_chunk(results, "c1", 1)  # planned 2 -> short
    with pytest.raises(pub.PublishError, match="short chunk"):
        pub.publish(results, tmp_path / "out", {"c1": 0}, PLAN)


def test_publish_refuses_over_long_chunk(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _write_chunk(results, "c1", 3)  # planned 2 -> over
    with pytest.raises(pub.PublishError, match="over-long chunk"):
        pub.publish(results, tmp_path / "out", {"c1": 0}, PLAN)


def test_publish_refuses_duplicate_chunk(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _write_chunk(results, "c1", 2)
    _write_chunk(results, "c1_retry", 2)   # both exit-0 and full -> duplicate
    with pytest.raises(pub.PublishError, match="duplicate chunk"):
        pub.publish(results, tmp_path / "out", {"c1": 0, "c1_retry": 0}, PLAN)


def test_publish_refuses_missing_chunk(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _write_chunk(results, "c1", 2)   # present and full, but...
    with pytest.raises(pub.PublishError, match="missing chunk"):
        pub.publish(results, tmp_path / "out", {"c1": 1}, PLAN)  # ...exit code != 0


def test_retry_dir_is_selected_when_original_failed(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _write_chunk(results, "c1", 1)          # original: short + failed
    _write_chunk(results, "c1_retry", 2)    # retry: full + exit 0
    out = tmp_path / "out"
    pub.publish(results, out, {"c1": 1, "c1_retry": 0}, PLAN)
    # the published records come from the retry (2 records).
    assert len((out / "c1" / "records.jsonl").read_text().splitlines()) == 2


def test_parse_status_log(tmp_path):
    log = tmp_path / "status.log"
    log.write_text("c1 exit_code=0\nc2 exit_code=1\n", encoding="utf-8")
    assert pub.parse_status_log(log) == {"c1": 0, "c2": 1}


def test_build_plan_counts_match_section_9():
    import importlib.util as u
    spec = u.spec_from_file_location("phase3b_run", REPO_ROOT / "phase3b" / "run.py")
    run = u.module_from_spec(spec); sys.modules["phase3b_run"] = run; spec.loader.exec_module(run)
    manifest = __import__("harness").load_manifest(REPO_ROOT / "phase3b" / "data")
    plan = pub.build_plan(manifest)
    agent = sum(p["expected_count"] for p in plan if p["kind"] == "agent")
    trans = sum(p["expected_count"] for p in plan if p["kind"] == "transcription")
    assert agent == 960 and trans == 400
