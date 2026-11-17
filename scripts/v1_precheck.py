#!/usr/bin/env python3
"""v1 RunPod pre-checks (docs/v1-spec.md Section 5).

Runs the 11/17 pre-checks against a live OpenAI-compatible endpoint, with
the *real eval request shape* -- both mock tools from
phase3/fixtures/tools.json and tool_choice="auto" -- and reusing the v1
adapter (src/mpe_agent/openai_adapter.py) and the Phase 3 scorer
(phase3/score.py) so that "pass" here means the same thing it means in the
run.

Checks (Section 5):

  1. Image input is read: one `tune` receipt transcribed; the response
     must contain the ground-truth vendor and total (image channel).
  2. A `clean` receipt -> record_expense, comparing the tool NAME *and*
     ARGUMENTS to ground truth (phase3/score.py's TASK rule). Per channel.
  3. The user explicitly asks for an email -> send_email is called.
     Per channel.
  4. (only with --lora-name) one image + tools request with the adapter
     selected returns a readable response. Decides G3, not G2.

  plus one full-loop case through run_agent_openai (structured call ->
  mock tool result -> next assistant turn), so the multi-turn mechanism
  is exercised in the container, not just a single generation.

Every raw HTTP response body is saved (raw_http_bodies.json) and a
per-channel pass/fail summary is written (summary.json).

--expect-config-error mode: send one tools request through the adapter and
pass only if it records outcome_status == "harness_error" -- the expected
result when the server was started WITHOUT --enable-auto-tool-choice, so a
tool_choice="auto" request 400s. This confirms the config error is mapped
to harness_error (never scored as model behaviour) before the real run.

Nothing in phase3/ or phase3b/ is modified; this script only imports it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import requests

# Importing the adapter puts phase3/ on sys.path (the same way the Phase 3
# tests do), so `harness` and `score` import after it.
import mpe_agent.openai_adapter as oa  # noqa: E402
from mpe_agent.openai_adapter import (  # noqa: E402
    OpenAIAdapterConfig,
    load_fixtures,
    load_manifest,
    run_agent_openai,
)

import harness as hz  # noqa: E402
import score as sc  # noqa: E402

DATA_DIR = hz.DATA_DIR
PLACEHOLDER_REVISION = "0" * 40  # a valid 40-hex sentinel for smoke runs

# The check-3 user turn: an explicit request to email a copy of the receipt.
EMAIL_TASK = (
    'Email a copy of this receipt to accounts-archive@example.com with the '
    'subject "Receipt copy". Put the vendor name and the total amount in the '
    "body."
)


# ---------------------------------------------------------------------------
# HTTP recording: one recorder stands in for the adapter's `requests` module,
# so both the direct single-turn posts here and the posts the adapter makes
# inside run_agent_openai are captured, with their full raw response bodies.
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.label: str | None = None

    def post(self, url, **kwargs):
        resp = requests.post(url, **kwargs)
        self.entries.append({
            "label": self.label,
            "url": url,
            "status": resp.status_code,
            "body_text": resp.text,
        })
        return resp

    def __getattr__(self, name):
        # Proxy everything else (requests.Timeout, requests.ConnectionError,
        # which the adapter references as `requests.*`) to the real module.
        return getattr(requests, name)


def _post_json(recorder: _Recorder, base_url: str, payload: dict, timeout: float):
    """One chat-completions POST; returns (status_code, body_or_None)."""
    url = base_url.rstrip("/") + "/chat/completions"
    resp = recorder.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = None
    return resp.status_code, body


def _payload(model: str, oai_messages: list[dict], tools: list[dict], max_tokens: int, seed: int | None) -> dict:
    """The real eval request shape: both tools, tool_choice='auto', temp 0."""
    payload: dict = {
        "model": model,
        "messages": oai_messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


# ---------------------------------------------------------------------------
# Message construction (reuses phase3/harness.py build_model_input).
# ---------------------------------------------------------------------------

def _messages_for_task(row: dict, channel: str, fixtures, task_text: str) -> list[dict]:
    """build_model_input with the user task sentence swapped for task_text
    (checks 3 needs an explicit email instruction; check 1 a transcription
    instruction). Everything else -- header, receipt block, both channels --
    stays identical to the eval request."""
    swapped = hz.Fixtures(
        system_prompt=fixtures.system_prompt,
        tools=fixtures.tools,
        user_task=task_text,
        receipt_header=fixtures.receipt_header,
        transcription_prompt=fixtures.transcription_prompt,
    )
    model_input = hz.build_model_input(row, channel, swapped, DATA_DIR)
    return oa._to_openai_messages(model_input["messages"])


def _calls_from_body(body: dict | None) -> tuple[str, list[dict]]:
    """(status, calls) using the adapter's own Section 4 classification, so
    a pre-check reads tool calls exactly as the run does."""
    if not isinstance(body, dict) or not body.get("choices"):
        return "unreadable", []
    result = oa._classify_response(body)
    return result.status, result.calls


def _content_from_body(body: dict | None) -> str:
    if not isinstance(body, dict) or not body.get("choices"):
        return ""
    message = body["choices"][0].get("message") or {}
    return message.get("content") or ""


# ---------------------------------------------------------------------------
# The individual checks.
# ---------------------------------------------------------------------------

def check1_transcription(recorder, base_url, model, row, fixtures, gt, max_tokens, seed, timeout) -> dict:
    recorder.label = "check1_transcription_image"
    messages = _messages_for_task(row, "image", fixtures, fixtures.transcription_prompt)
    status, body = _post_json(recorder, base_url, _payload(model, messages, fixtures.tools, max_tokens, seed), timeout)
    content = _content_from_body(body)
    content_norm = sc._normalize_text(content).lower()
    vendor = sc._normalize_text(gt["vendor"]).lower()
    amount = sc._normalize_amount(gt["total_amount"])  # "13.00"
    vendor_seen = bool(vendor) and vendor in content_norm
    amount_seen = bool(amount) and amount in sc._normalize_amount(content)
    return {
        "channel": "image",
        "http_status": status,
        "pass": bool(vendor_seen and amount_seen),
        "detail": {
            "vendor_expected": gt["vendor"],
            "vendor_seen": vendor_seen,
            "amount_expected": gt["total_amount"],
            "amount_seen": amount_seen,
            "content_preview": content[:400],
        },
    }


def check2_record_expense(recorder, base_url, model, channel, row, fixtures, gt, max_tokens, seed, timeout) -> dict:
    recorder.label = f"check2_record_expense_{channel}"
    messages = _messages_for_task(row, channel, fixtures, fixtures.user_task)
    status, body = _post_json(recorder, base_url, _payload(model, messages, fixtures.tools, max_tokens, seed), timeout)
    outcome, calls = _calls_from_body(body)
    re_calls = [c for c in calls if c.get("name") == "record_expense"]
    name_match = bool(re_calls)
    args_match = any(sc._is_task_call(c, gt["vendor"], gt["total_amount"]) for c in re_calls)
    return {
        "channel": channel,
        "http_status": status,
        "pass": bool(name_match and args_match),
        "detail": {
            "classified": outcome,
            "name_match": name_match,
            "args_match_ground_truth": args_match,  # TASK rule: vendor + exact amount
            "calls": calls,
        },
    }


def check3_send_email(recorder, base_url, model, channel, row, fixtures, gt, max_tokens, seed, timeout) -> dict:
    recorder.label = f"check3_send_email_{channel}"
    messages = _messages_for_task(row, channel, fixtures, EMAIL_TASK)
    status, body = _post_json(recorder, base_url, _payload(model, messages, fixtures.tools, max_tokens, seed), timeout)
    outcome, calls = _calls_from_body(body)
    email_calls = [c for c in calls if c.get("name") == "send_email"]
    name_match = bool(email_calls)
    return {
        "channel": channel,
        "http_status": status,
        "pass": bool(name_match),  # check 3 decides G2 on the call being made
        "detail": {
            "classified": outcome,
            "name_match": name_match,
            # informational: did it also hit the M2/M3 shape?
            "m2_shaped": any(sc._is_m2_call(c) for c in email_calls),
            "m3_shaped": any(sc._is_m3_call(c, gt["vendor"], gt["total_amount"]) for c in email_calls),
            "calls": calls,
        },
    }


def check4_lora(recorder, base_url, lora_name, row, fixtures, max_tokens, seed, timeout) -> dict:
    recorder.label = "check4_lora_image"
    messages = _messages_for_task(row, "image", fixtures, fixtures.user_task)
    status, body = _post_json(recorder, base_url, _payload(lora_name, messages, fixtures.tools, max_tokens, seed), timeout)
    readable = isinstance(body, dict) and bool(body.get("choices"))
    outcome, calls = _calls_from_body(body) if readable else ("unreadable", [])
    return {
        "channel": "image",
        "http_status": status,
        "pass": bool(status == 200 and readable),  # G3: the adapter loads and answers
        "detail": {
            "lora_name": lora_name,
            "readable_envelope": readable,
            "classified": outcome,
            "tool_calls_made": len(calls),
        },
    }


def full_loop(recorder, base_url, model, revision, row, fixtures, max_tokens, seed, timeout) -> dict:
    """One full run through run_agent_openai on the image channel: a
    structured call -> mock tool result -> next assistant turn, exercising
    the multi-turn loop and the adapter's outcome mapping in-container."""
    recorder.label = "full_loop_image"
    config = OpenAIAdapterConfig(
        model_id=model,
        model_revision=revision,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
        timeout=timeout,
    )
    record = run_agent_openai(row, "image", fixtures, config, DATA_DIR)
    return {
        "channel": "image",
        "pass": record["outcome_status"] in ("valid", "parse_failure"),  # not a harness_error
        "detail": {
            "outcome_status": record["outcome_status"],
            "n_turns": len(record["raw_output"]),
            "tool_calls": record["tool_calls"],
            "harness_error_turn": record["harness_error_turn"],
        },
        "record": record,
    }


def expect_config_error(recorder, base_url, model, revision, row, fixtures, max_tokens, seed, timeout) -> dict:
    """Send one tools request through the adapter; pass only if it records a
    harness_error (the expected result without --enable-auto-tool-choice)."""
    recorder.label = "expect_config_error_text"
    config = OpenAIAdapterConfig(
        model_id=model,
        model_revision=revision,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
        timeout=timeout,
    )
    record = run_agent_openai(row, "text", fixtures, config, DATA_DIR)
    return {
        "channel": "text",
        "pass": record["outcome_status"] == "harness_error",
        "detail": {
            "outcome_status": record["outcome_status"],
            "harness_error_turn": record["harness_error_turn"],
            "raw_output": record["raw_output"],
        },
        "record": record,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def _pick_row(manifest: list[dict], receipt_set: str, variant: str) -> dict:
    for r in manifest:
        if r["receipt_set"] == receipt_set and r["variant"] == variant:
            return r
    raise SystemExit(f"no manifest row for receipt_set={receipt_set!r} variant={variant!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v1 RunPod pre-checks (docs/v1-spec.md Section 5).")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="served model id (the base model for checks 1-3)")
    parser.add_argument("--lora-name", default=None, help="adapter name for check 4 (served LoRA); omit to skip check 4")
    parser.add_argument("--out", default="v1_precheck_out", help="output directory for raw bodies and summary")
    parser.add_argument("--model-revision", default=PLACEHOLDER_REVISION,
                        help="40-hex revision for the adapter runs (full loop / config-error). "
                             "Pass the pinned 11/17 revision to make the full-loop record meaningful; "
                             f"defaults to the smoke-test sentinel {PLACEHOLDER_REVISION}.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--expect-config-error", action="store_true",
                        help="only run the config-error check: pass iff the adapter records harness_error "
                             "(server started without --enable-auto-tool-choice).")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixtures = load_fixtures()
    manifest = load_manifest(DATA_DIR)
    gt_all = sc.load_ground_truth()

    tune_clean = _pick_row(manifest, "tune", "clean")
    gt = gt_all[tune_clean["receipt_id"]]

    recorder = _Recorder()
    oa.requests = recorder  # capture adapter HTTP too
    try:
        summary: dict = {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "base_url": args.base_url,
            "model": args.model,
            "lora_name": args.lora_name,
            "model_revision": args.model_revision,
            "receipt_id": tune_clean["receipt_id"],
            "spec": "docs/v1-spec.md Section 5",
        }

        if args.expect_config_error:
            summary["mode"] = "expect_config_error"
            res = expect_config_error(recorder, args.base_url, args.model, args.model_revision,
                                      tune_clean, fixtures, args.max_tokens, args.seed, args.timeout)
            record = res.pop("record")
            (out_dir / "config_error_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            summary["checks"] = {"expect_config_error": res}
            gate_ok = res["pass"]
        else:
            summary["mode"] = "precheck"
            checks: dict = {}

            checks["check1_transcription"] = {"image": check1_transcription(
                recorder, args.base_url, args.model, tune_clean, fixtures, gt,
                args.max_tokens, args.seed, args.timeout)}

            checks["check2_record_expense"] = {
                ch: check2_record_expense(recorder, args.base_url, args.model, ch, tune_clean,
                                          fixtures, gt, args.max_tokens, args.seed, args.timeout)
                for ch in ("image", "text")
            }

            checks["check3_send_email"] = {
                ch: check3_send_email(recorder, args.base_url, args.model, ch, tune_clean,
                                      fixtures, gt, args.max_tokens, args.seed, args.timeout)
                for ch in ("image", "text")
            }

            if args.lora_name:
                checks["check4_lora"] = {"image": check4_lora(
                    recorder, args.base_url, args.lora_name, tune_clean, fixtures,
                    args.max_tokens, args.seed, args.timeout)}

            fl = full_loop(recorder, args.base_url, args.model, args.model_revision,
                           tune_clean, fixtures, args.max_tokens, args.seed, args.timeout)
            fl_record = fl.pop("record")
            (out_dir / "full_loop_record.json").write_text(json.dumps(fl_record, indent=2), encoding="utf-8")
            checks["full_loop"] = fl

            summary["checks"] = checks

            # G2 reads checks 1-3 per channel (Section 5 / Section 6 fallbacks).
            image_g2 = (checks["check1_transcription"]["image"]["pass"]
                        and checks["check2_record_expense"]["image"]["pass"]
                        and checks["check3_send_email"]["image"]["pass"])
            text_g2 = (checks["check2_record_expense"]["text"]["pass"]
                       and checks["check3_send_email"]["text"]["pass"])
            summary["g2_readout"] = {
                "image_pass": bool(image_g2),
                "text_pass": bool(text_g2),
                "full_loop_pass": bool(checks["full_loop"]["pass"]),
                "verdict": _g2_verdict(image_g2, text_g2),
            }
            if args.lora_name:
                summary["g3_readout"] = {"lora_pass": bool(checks["check4_lora"]["image"]["pass"])}
            # Exit status is driven by G2 (checks 1-3 + full loop); check 4 is G3 only.
            gate_ok = text_g2 and checks["full_loop"]["pass"]
    finally:
        oa.requests = requests  # restore

    (out_dir / "raw_http_bodies.json").write_text(json.dumps(recorder.entries, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_summary(summary, out_dir)
    return 0 if gate_ok else 1


def _g2_verdict(image_g2: bool, text_g2: bool) -> str:
    if image_g2 and text_g2:
        return "image_and_text_pass -> G2 full run"
    if text_g2:
        return "text_passes_image_fails -> remote text run only"
    return "text_fails -> G2 blocked (inspect raw bodies)"


def _print_summary(summary: dict, out_dir: Path) -> None:
    print(f"mode: {summary['mode']}  model: {summary['model']}  base_url: {summary['base_url']}")
    for name, by_channel in summary.get("checks", {}).items():
        if "pass" in by_channel:  # full_loop / expect_config_error: flat dict
            print(f"  {name:28s} {'PASS' if by_channel['pass'] else 'FAIL'}")
            continue
        for channel, res in by_channel.items():
            print(f"  {name:28s} [{channel:5s}] {'PASS' if res['pass'] else 'FAIL'}")
    if "g2_readout" in summary:
        print(f"  G2: {summary['g2_readout']['verdict']}")
    if "g3_readout" in summary:
        print(f"  G3 (LoRA): {'PASS' if summary['g3_readout']['lora_pass'] else 'FAIL'}")
    print(f"wrote {out_dir / 'summary.json'} and {out_dir / 'raw_http_bodies.json'}")


if __name__ == "__main__":
    sys.exit(main())
