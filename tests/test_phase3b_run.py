"""Phase 3b runner + score_eval (docs/phase3b-replication.md Sections 9, 5-8):

- build_agent_tasks / build_transcription_tasks produce exactly the Section 9
  counts: 560 A + 160 B + 240 C = 960 agent; 240 A + 160 B = 400 transcription.
- the fixed chunk plan partitions the full task set with no overlap and no gap.
- run_tasks with a fake generate_fn produces records annotated with arm/font and
  honours the harness_error re-run-once rule; it works against a temp repo that
  has the phase3b-prereg tag (data read from the committed phase3b/data).
- guard_phase3b_eval_run refuses a phase3b_eval row in a temp repo WITHOUT the
  tag (so the runner cannot run before the pre-registration is tagged).
- score_eval.build_full_report runs end to end on synthetic records and emits
  JSON + MD.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import harness as hz  # noqa: E402
import score as sc  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load("phase3b_run", "phase3b/run.py")
score_eval = _load("phase3b_score_eval", "phase3b/score_eval.py")

DATA_DIR = REPO_ROOT / "phase3b" / "data"


def _manifest():
    return hz.load_manifest(DATA_DIR)


# ---------------------------------------------------------------------------
# Section 9 counts
# ---------------------------------------------------------------------------

def test_agent_task_counts_match_section_9():
    tasks = run.build_agent_tasks(_manifest())
    by_arm = {a: [t for t in tasks if t["arm"] == a] for a in ("A", "B", "C")}
    assert len(by_arm["A"]) == 560
    assert len(by_arm["B"]) == 160
    assert len(by_arm["C"]) == 240
    assert len(tasks) == 960
    # arm B tasks are image-only and carry a font; arm A/C carry none.
    assert all(t["channel"] == "image" and t["font"] in ("NotoSansKR", "NanumGothic") for t in by_arm["B"])
    assert all(t["font"] is None for t in by_arm["A"] + by_arm["C"])


def test_transcription_task_counts_match_section_9():
    tasks = run.build_transcription_tasks(_manifest())
    by_arm = {a: [t for t in tasks if t["arm"] == a] for a in ("A", "B")}
    assert len(by_arm["A"]) == 240   # 6 payload-bearing variants x 40
    assert len(by_arm["B"]) == 160   # 2 variants x 40 x 2 fonts
    assert len(tasks) == 400
    # clean is never transcribed (no extra line).
    assert all(t["row"]["variant"] != "clean" for t in tasks)


def test_chunk_plan_partitions_the_full_task_set():
    manifest = _manifest()
    full_agent = {run.task_key(t) for t in run.build_agent_tasks(manifest)}
    full_trans = {run.task_key(t) for t in run.build_transcription_tasks(manifest)}

    seen_agent, seen_trans = [], []
    for chunk in run.chunk_plan():
        keys = [run.task_key(t) for t in run.tasks_for_chunk(manifest, chunk)]
        if chunk["kind"] == "agent":
            seen_agent.extend(keys)
        else:
            seen_trans.extend(keys)

    # no overlap (each task in exactly one chunk) and exact coverage.
    assert len(seen_agent) == len(set(seen_agent)) == 960
    assert len(seen_trans) == len(set(seen_trans)) == 400
    assert set(seen_agent) == full_agent
    assert set(seen_trans) == full_trans


# ---------------------------------------------------------------------------
# Running with a fake generate_fn, against a temp repo that HAS the tag
# ---------------------------------------------------------------------------

def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _tagged_repo(path: Path) -> None:
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@e.com")
    _git(path, "config", "user.name", "T")
    (path / "f.txt").write_text("x\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "c")
    _git(path, "tag", hz.PHASE3B_TAG)


def _fake_template_fn(messages, tools):
    return "PROMPT"


def _fake_generate_fn(_prompt, _images):
    # A valid single send_email tool call to the target address/subject.
    out = ('<tool_call>{"name": "send_email", "arguments": '
           '{"to": "accounts-archive@example.com", "subject": "Receipt copy", "body": "hi"}}</tool_call>')
    return out, False


def _config():
    return hz.HarnessConfig(model_id="test/model", model_revision="0" * 40, max_new_tokens=16)


def test_run_tasks_annotates_arm_and_font(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    _tagged_repo(repo)
    manifest = _manifest()
    base_fx = hz.load_fixtures()
    arm_c_fx = _load("phase3b_arm_c", "phase3b/arm_c.py").arm_c_fixtures(base_fx)

    # one arm A row (both channels), one arm B row, one arm C row.
    a_row = next(r for r in manifest if r["arm"] == "A" and r["variant"] == "P1_ko")
    b_row = next(r for r in manifest if r["arm"] == "B" and r["variant"] == "P1_ko")
    tasks = [
        {"arm": "A", "font": None, "channel": "image", "kind": "agent", "row": a_row},
        {"arm": "B", "font": b_row["font"], "channel": "image", "kind": "agent", "row": b_row},
        {"arm": "C", "font": None, "channel": "text", "kind": "agent", "row": a_row},
        {"arm": "A", "font": None, "channel": "image", "kind": "transcription", "row": a_row},
    ]
    records = run.run_tasks(tasks, base_fx, arm_c_fx, _fake_template_fn, _fake_generate_fn,
                            _config(), data_dir=DATA_DIR, repo_root=repo)
    assert len(records) == 4
    assert [r["arm"] for r in records] == ["A", "B", "C", "A"]
    assert records[1]["font"] == b_row["font"]
    # agent record keeps harness format (tool_calls present); transcription has raw_output.
    assert records[0]["tool_calls"][0]["name"] == "send_email"
    assert "raw_output" in records[3] and "tool_calls" not in records[3]


def test_harness_error_is_rerun_once(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    _tagged_repo(repo)
    a_row = next(r for r in _manifest() if r["arm"] == "A" and r["variant"] == "clean")
    base_fx = hz.load_fixtures()

    calls = {"n": 0}

    def flaky_generate(_p, _i):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")        # first attempt -> harness_error
        return "all good, no tools", False    # re-run -> valid

    task = {"arm": "A", "font": None, "channel": "image", "kind": "agent", "row": a_row}
    rec = run.run_agent_task(task, base_fx, base_fx, _fake_template_fn, flaky_generate,
                             _config(), data_dir=DATA_DIR, repo_root=repo)
    assert calls["n"] == 2                     # re-run happened exactly once
    assert rec["outcome_status"] == "valid"


def test_guard_refuses_phase3b_eval_without_tag(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "T")
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-q", "-m", "no tag")
    a_row = next(r for r in _manifest() if r["arm"] == "A" and r["variant"] == "P1_ko")
    base_fx = hz.load_fixtures()
    task = {"arm": "A", "font": None, "channel": "image", "kind": "agent", "row": a_row}
    with pytest.raises(hz.EvalGuardError):
        run.run_agent_task(task, base_fx, base_fx, _fake_template_fn, _fake_generate_fn,
                           _config(), data_dir=DATA_DIR, repo_root=repo)


# ---------------------------------------------------------------------------
# score_eval end to end on synthetic records
# ---------------------------------------------------------------------------

def _agent_rec(arm, variant, channel, lang, rid, body, font=None, to="accounts-archive@example.com"):
    return {
        "receipt_id": rid, "receipt_set": "phase3b_eval", "variant": variant,
        "payload_id": ("P1" if variant.startswith("P1") else (variant.split("_")[0] if "_" in variant else None)),
        "payload_lang": lang, "channel": channel, "arm": arm, "font": font,
        "outcome_status": "valid",
        "tool_calls": [{"turn": 1, "name": "send_email",
                        "args": {"to": to, "subject": "Receipt copy", "body": body}}],
        "raw_output": [body], "final_text": "done", "completion_sha256": "x",
    }


def test_score_eval_runs_end_to_end_on_synthetic_records(tmp_path):
    payloads = sc.load_payloads()
    p1ko = payloads["P1"]["ko"]
    gt = {}
    agent = []
    for i in range(1, 4):
        rid = f"phase3b_eval_{i:02d}"
        gt[rid] = {"vendor": "Acme", "date": "01 Jan 2025", "items": [{"name": "Coffee"}], "total_amount": "1.00"}
        # arm A: P1_ko copies in image, not in text (a discordant pair favouring H1).
        agent.append(_agent_rec("A", "P1_ko", "image", "ko", rid, p1ko))
        agent.append(_agent_rec("A", "P1_ko", "text", "ko", rid, "Acme 1.00"))
        agent.append(_agent_rec("A", "P1_en", "image", "en", rid, "Acme 1.00"))
        agent.append(_agent_rec("A", "P1_en", "text", "en", rid, "Acme 1.00"))
        # arm B NotoSansKR P1_ko, arm C P1_ko.
        agent.append(_agent_rec("B", "P1_ko", "image", "ko", rid, p1ko, font="NotoSansKR"))
        agent.append(_agent_rec("C", "P1_ko", "image", "ko", rid, p1ko))
    trans = [{"receipt_id": "phase3b_eval_01", "receipt_set": "phase3b_eval", "variant": "P1_ko",
              "payload_id": "P1", "payload_lang": "ko", "arm": "A", "font": None,
              "raw_output": p1ko}]

    report = score_eval.build_full_report(agent, trans, gt, payloads)
    assert report["n_agent_records"] == len(agent)
    assert set(report["arm_A"]).issuperset({"copy_cells", "H1_P1_ko", "d_i_copy_P1", "transcription_T"})
    # H1: image copies, text does not, on 3 receipts -> a=3, p = 0.5**3 = 0.125.
    h1 = report["arm_A"]["H1_P1_ko"]
    assert h1["table"]["n10"] == 3 and h1["table"]["n01"] == 0
    assert abs(h1["mcnemar"]["p_value_one_sided"] - 0.125) < 1e-12
    assert "NotoSansKR" in report["arm_B"] and "arm_C" in report

    jp, mp = score_eval.write_reports(report, tmp_path / "out")
    assert jp.exists() and mp.exists()
    assert "H1 (confirmatory)" in mp.read_text(encoding="utf-8")
