"""phase4/checks.py must EXIT NON-ZERO on any failure. Inject a bad vendor
overlap, an over-length example and a broken loss mask, and assert the command
fails. Needs the Qwen3-VL-4B tokenizer -> marked `network`."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4"))
CHECKS = REPO / "phase4" / "checks.py"
CONFIG = REPO / "phase4" / "train_config.yaml"
TRAIN = REPO / "phase4" / "data" / "train" / "train.jsonl"


def _examples_by_type():
    out = {}
    for line in open(TRAIN, encoding="utf-8"):
        e = json.loads(line)
        out.setdefault(e["type"], e)
        if len(out) == 3:
            break
    return out


def _write(tmp_path, examples):
    p = tmp_path / "train.jsonl"
    p.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    return p


def _run_checks(train_file) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECKS), "--train-file", str(train_file), "--config", str(CONFIG)],
                          capture_output=True, text=True)


@pytest.mark.network
def test_passes_on_real_data():
    r = _run_checks(TRAIN)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.network
def test_fails_on_vendor_overlap(tmp_path):
    ex = _examples_by_type()
    bad = json.loads(json.dumps(ex["injected"]))
    # set the record_expense vendor to a phase3 VENDORS name (forbidden in training)
    for m in bad["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            m["tool_calls"][0]["function"]["arguments"]["vendor"] = "Riverside Hardware"
            break
    p = _write(tmp_path, [bad, ex["clean"], ex["authorised_email"]])
    r = _run_checks(p)
    assert r.returncode != 0 and "overlap" in r.stdout.lower()


@pytest.mark.network
def test_fails_on_over_length(tmp_path):
    ex = _examples_by_type()
    bad = json.loads(json.dumps(ex["clean"]))
    bad["messages"][1]["content"] += " " + ("filler " * 500)   # push well past max_seq_len=680
    p = _write(tmp_path, [bad])
    r = _run_checks(p)
    assert r.returncode != 0 and "max_seq_len" in r.stdout


@pytest.mark.network
def test_fails_on_broken_mask(tmp_path, monkeypatch):
    import checks
    tiny = _write(tmp_path, list(_examples_by_type().values()))

    def broken(messages, tok, tools):
        s = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=False)
        ids = tok(s, add_special_tokens=False)["input_ids"]
        return ids, [-100] * len(ids), []        # mask EVERYTHING -> gold args fall outside loss

    monkeypatch.setattr(checks, "assistant_token_labels", broken)
    failures = checks.run_checks(tiny, CONFIG, verbose=False)
    assert any("gold tool-call arg" in f for f in failures)
    assert checks.main(["--train-file", str(tiny), "--config", str(CONFIG)]) == 1  # command exits non-zero
