"""Regression: build_model_input gains an optional per-row user_task, but when
the row has no user_task (every existing manifest), the built request is
byte-identical to before. Covers the Phase 3, Phase 3b and v1 manifests. Also
checks a present user_task overrides the task for that row only.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import harness as hz

REPO = Path(__file__).resolve().parent.parent
FIX = hz.load_fixtures()

MANIFESTS = [
    (REPO / "phase3" / "data" / "manifest.jsonl", REPO / "phase3" / "data"),
    (REPO / "phase3b" / "data" / "manifest.jsonl", REPO / "phase3b" / "data"),
    (REPO / "runs" / "v1_manifest.jsonl", REPO / "phase3b" / "data"),
]


def _fp(messages):
    """Canonical fingerprint of a built request; image blocks -> sha of pixels."""
    def norm(block):
        if block.get("type") == "image":
            return {"type": "image", "sha": hashlib.sha256(block["image"].tobytes()).hexdigest()}
        return block
    out = []
    for m in messages:
        c = m["content"]
        c = [norm(b) for b in c] if isinstance(c, list) else c
        out.append({"role": m["role"], "content": c})
    return json.dumps(out, sort_keys=True, ensure_ascii=False)


def _reference_old_request(row, channel, data_dir):
    """Independent reconstruction of the PRE-change build_model_input (task always
    from fixtures.user_task). Byte-identity to this proves nothing changed."""
    if channel == "image":
        receipt_block = {"type": "image", "image": hz.load_image(data_dir, row["image_path"])}
    else:
        text = (data_dir / row["transcript_path"]).read_text(encoding="utf-8")
        receipt_block = {"type": "text", "text": text + "\n"}
    return [
        {"role": "system", "content": [{"type": "text", "text": FIX.system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": FIX.user_task + "\n"},
            {"type": "text", "text": FIX.receipt_header + "\n"},
            receipt_block]},
    ]


def _rows(manifest, n=4):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    return rows[:n]


def test_byte_identical_when_user_task_absent():
    checked = 0
    for manifest, data_dir in MANIFESTS:
        for row in _rows(manifest):
            assert "user_task" not in row
            for channel in ("image", "text"):
                new = hz.build_model_input(row, channel, FIX, data_dir)["messages"]
                old = _reference_old_request(row, channel, data_dir)
                assert _fp(new) == _fp(old), f"request changed for {manifest.name} {row['receipt_id']} {channel}"
                checked += 1
    assert checked == len(MANIFESTS) * 4 * 2


def test_null_user_task_is_unchanged():
    manifest, data_dir = MANIFESTS[0]
    row = dict(_rows(manifest, 1)[0]); row["user_task"] = None
    new = hz.build_model_input(row, "text", FIX, data_dir)["messages"]
    old = _reference_old_request(row, "text", data_dir)
    assert _fp(new) == _fp(old)


def test_present_user_task_overrides_only_the_task():
    manifest, data_dir = MANIFESTS[1]
    base = _rows(manifest, 1)[0]
    custom = "Record this and email a copy to team@example.test with the subject Receipt Please."
    row = dict(base); row["user_task"] = custom
    msgs = hz.build_model_input(row, "text", FIX, data_dir)["messages"]
    user = msgs[1]["content"]
    assert user[0]["text"] == custom + "\n"                       # task replaced
    assert user[1]["text"] == FIX.receipt_header + "\n"           # header unchanged
    # receipt block unchanged vs the no-override build
    plain = hz.build_model_input(base, "text", FIX, data_dir)["messages"][1]["content"]
    assert user[2] == plain[2]
    assert user[0]["text"] != plain[0]["text"]
