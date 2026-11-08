#!/usr/bin/env python3
"""Channel sanity check for Phase 3's R_eval records: did the image
channel actually receive an image (vision placeholder tokens, no
payload text), and did the text channel actually receive text (the
payload text, no vision tokens)? Descriptive, read-only; computes no
scoring measure and does not modify score.py, report.py,
report/phase3_results/, or the exploratory_m3/failure_cases files.

Verifies report/phase3_eval/SHA256SUMS before reading any record. Every
number comes from the committed records in report/phase3_eval/ and
phase3/fixtures/payloads.json. Writes report/phase3_exploratory/channel_check.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import score as sc  # noqa: E402

EVAL_DIR = REPO_ROOT / "report" / "phase3_eval"
OUT_DIR = REPO_ROOT / "report" / "phase3_exploratory"

AGENT_CHUNKS = [f"chunk_{i:02d}" for i in range(1, 9)]
VISION_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"


def verify_sha256sums(eval_dir: Path) -> None:
    sums_path = eval_dir / "SHA256SUMS"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel_path = line.split(maxsplit=1)
        actual = hashlib.sha256((eval_dir / rel_path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"sha256 mismatch for {rel_path}: expected {expected}, got {actual}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_all_agent_records() -> list[dict]:
    records: list[dict] = []
    for chunk in AGENT_CHUNKS:
        records.extend(load_jsonl(EVAL_DIR / chunk / "records.jsonl"))
    return records


def payload_text_for(record: dict, payloads: dict) -> str | None:
    if record["payload_id"] is None:
        return None
    return payloads[record["payload_id"]][record["payload_lang"]]


def main() -> None:
    verify_sha256sums(EVAL_DIR)
    records = load_all_agent_records()
    assert len(records) == 200, f"expected 200 agent records, got {len(records)}"

    payloads = sc.load_payloads()

    image_total = image_vision_present = image_vision_missing = 0
    image_with_payload = image_payload_leak = image_payload_absent = 0
    text_total = text_vision_present_bug = text_vision_absent = 0
    text_with_payload = text_payload_present = text_payload_missing = 0

    leak_examples: list[dict] = []
    missing_examples: list[dict] = []

    for record in records:
        prompt0 = record["rendered_prompts"][0]
        payload_text = payload_text_for(record, payloads)

        if record["channel"] == "image":
            image_total += 1
            if VISION_TOKENS in prompt0:
                image_vision_present += 1
            else:
                image_vision_missing += 1
            if payload_text is not None:
                image_with_payload += 1
                normalized_prompt = " ".join(prompt0.split())
                normalized_payload = " ".join(payload_text.split())
                if payload_text in prompt0 or normalized_payload in normalized_prompt:
                    image_payload_leak += 1
                    leak_examples.append({"receipt_id": record["receipt_id"], "variant": record["variant"]})
                else:
                    image_payload_absent += 1
        else:  # text
            text_total += 1
            if VISION_TOKENS in prompt0:
                text_vision_present_bug += 1
            else:
                text_vision_absent += 1
            if payload_text is not None:
                text_with_payload += 1
                normalized_prompt = " ".join(prompt0.split())
                normalized_payload = " ".join(payload_text.split())
                if normalized_payload in normalized_prompt:
                    text_payload_present += 1
                else:
                    text_payload_missing += 1
                    missing_examples.append({"receipt_id": record["receipt_id"], "variant": record["variant"]})

    report = {
        "n_agent_records": len(records),
        "image_channel": {
            "n_records": image_total,
            "vision_tokens_present": image_vision_present,
            "vision_tokens_missing": image_vision_missing,
            "n_records_with_payload": image_with_payload,
            "payload_text_leaked_into_prompt": image_payload_leak,
            "payload_text_correctly_absent": image_payload_absent,
            "leak_examples": leak_examples,
        },
        "text_channel": {
            "n_records": text_total,
            "vision_tokens_present_bug_check": text_vision_present_bug,
            "vision_tokens_absent_as_expected": text_vision_absent,
            "n_records_with_payload": text_with_payload,
            "payload_text_present": text_payload_present,
            "payload_text_missing": text_payload_missing,
            "missing_examples": missing_examples,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "channel_check.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
