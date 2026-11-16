"""OpenAI-compatible endpoint adapter for the agent-injection eval
(docs/v1-spec.md Sections 3, 4).

Phase 3 served the model in transformers and parsed `<tool_call>` text
blocks from the raw output (phase3/harness.py). v1 reads structured
`tool_calls` from an OpenAI-compatible chat-completions response. This
module keeps everything else identical to phase3/harness.py by importing
and reusing it -- the fixtures, the model-input construction, the eval
guards, the fixed mock tool results, the 3-turn loop shape, the
tool-call bookkeeping and the Section 9 record schema -- and changes only
the outcome mapping (Section 4):

    Phase 3                              -> v1
    no <tool_call> block                 -> tool_calls empty and no
                                            call-like JSON in content
                                            (valid, zero calls)
    malformed <tool_call> JSON           -> arguments is not valid JSON,
                                            or the server returns a tool-
                                            parser error (parse_failure)
    untagged call-like JSON              -> tool_calls empty but content
                                            holds name+arguments JSON
                                            (parse_failure; reuse
                                            contains_untagged_tool_call_json)
    OOM / crash                          -> timeout, HTTP 5xx or a
                                            connection error after retries,
                                            or a non-tool-parser 4xx (401,
                                            404, 400 context length)
                                            (harness_error)

The multi-turn scoring rule is unchanged: tool calls from valid turns
before a parse_failure stay on the record (phase3/harness.py's loop and
phase3/score.py both rely on this).

Every record additionally carries the v1 reporting fields (Section 3):
backend and version, temperature, seed, max tokens, latency, token
counts and the run's retry count, alongside model_id/model_revision.

No model is loaded here; the only external dependency is an HTTP endpoint
speaking the OpenAI chat-completions schema (vLLM's server, or a mock).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# phase3/harness.py is imported, never modified: it is added to sys.path the
# same way the Phase 3 tests import it (there is no phase3 package to install).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHASE3_DIR = _REPO_ROOT / "phase3"
if str(_PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE3_DIR))

import harness as _hz  # noqa: E402

# Re-exported so callers (and tests) use one import site for the shared pieces.
REPO_ROOT = _hz.REPO_ROOT
DATA_DIR = _hz.DATA_DIR
CHANNELS = _hz.CHANNELS
MAX_ASSISTANT_TURNS = _hz.MAX_ASSISTANT_TURNS
MOCK_TOOL_RESULTS = _hz.MOCK_TOOL_RESULTS
Fixtures = _hz.Fixtures
load_fixtures = _hz.load_fixtures
load_manifest = _hz.load_manifest


@dataclass(frozen=True)
class OpenAIAdapterConfig:
    model_id: str
    model_revision: str
    base_url: str
    max_tokens: int
    temperature: float = 0.0
    seed: int | None = None
    backend: str = "openai-compatible"
    backend_version: str | None = None
    tool_choice: str = "auto"
    timeout: float = 60.0
    # Retries apply only to transient failures (timeout / 5xx / connection),
    # per Section 4: harness_error is "... after retries".
    max_retries: int = 2
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Message conversion: phase3/harness.py content-block messages -> OpenAI form.
# ---------------------------------------------------------------------------

def _image_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _to_openai_content(blocks: list[dict]) -> list[dict]:
    parts: list[dict] = []
    for block in blocks:
        if block["type"] == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image":
            parts.append({"type": "image_url", "image_url": {"url": _image_data_url(block["image"])}})
    return parts


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": _to_openai_content(m["content"])} for m in messages]


def _fingerprint_data_url(url: str) -> str:
    """Replaces the base64 payload of an image data URL with its sha256, so
    a stored rendered prompt stays small while remaining a reproducible
    fingerprint of the exact image bytes sent (same bytes -> same digest)."""
    prefix, sep, b64 = url.partition("base64,")
    if not sep or not prefix.startswith("data:"):
        return url
    digest = hashlib.sha256(b64.encode("ascii")).hexdigest()
    return f"{prefix}base64-sha256:{digest}"


def _render_payload(payload: dict) -> dict:
    """A copy of the request payload with every image base64 replaced by its
    sha256, used only for the stored rendered_prompts (Section 9). The
    request actually sent keeps the full base64."""
    messages = []
    for message in payload["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "image_url":
                    image_url = part["image_url"]
                    part = {**part, "image_url": {**image_url, "url": _fingerprint_data_url(image_url["url"])}}
                new_content.append(part)
            message = {**message, "content": new_content}
        messages.append(message)
    return {**payload, "messages": messages}


def _request_payload(config: OpenAIAdapterConfig, messages: list[dict], tools: list[dict]) -> dict:
    payload: dict[str, Any] = {
        "model": config.model_id,
        "messages": messages,
        "tools": tools,
        "tool_choice": config.tool_choice,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.seed is not None:
        payload["seed"] = config.seed
    return payload


def _assistant_echo(content: str, raw_tool_calls: list[dict]) -> dict:
    """The assistant message replayed into the next turn's history, so the
    endpoint sees its own tool_calls before the matching tool results."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if raw_tool_calls:
        message["tool_calls"] = raw_tool_calls
    return message


# ---------------------------------------------------------------------------
# HTTP call with transient-failure retries.
# ---------------------------------------------------------------------------

class _HarnessError(Exception):
    """A transient failure that survived every retry -> harness_error."""

    def __init__(self, message: str, retries: int) -> None:
        super().__init__(message)
        self.retries = retries


def _post_chat(config: OpenAIAdapterConfig, payload: dict) -> tuple[requests.Response, float, int]:
    """POSTs one chat-completions request. Retries only on a timeout,
    connection error or HTTP 5xx (up to config.max_retries), returning
    (response, latency_seconds, retries_used) for anything else -- a 2xx
    or a 4xx (the 4xx is classified downstream, not retried). Raises
    _HarnessError when a transient failure exhausts the retries."""
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    retries = 0
    while True:
        start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=config.timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if retries < config.max_retries:
                retries += 1
                continue
            raise _HarnessError(f"{type(exc).__name__}: {exc}", retries)
        latency = time.perf_counter() - start

        if response.status_code >= 500:
            if retries < config.max_retries:
                retries += 1
                continue
            raise _HarnessError(f"HTTP {response.status_code} after {retries} retries", retries)

        return response, latency, retries


# ---------------------------------------------------------------------------
# Response -> outcome mapping (Section 4).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TurnResult:
    status: str  # "valid" | "parse_failure" | "harness_error"
    calls: list[dict]  # [{"name", "args"}], parsed from structured tool_calls
    content: str
    raw_tool_calls: list[dict]  # echoed verbatim into the next turn
    finish_reason: str | None


# Substrings that identify an error envelope as coming from the server's
# tool-call parser (Section 4: only a tool-parser error is parse_failure).
_TOOL_PARSER_ERROR_SIGNALS = ("tool call", "tool_call", "tool-call", "tool parser", "toolparser")


def _error_text(body: dict) -> str:
    """Lower-cased text of an OpenAI-style error envelope ({"error": {...}}
    or {"error": "..."}), for classifying what the error is."""
    err = body.get("error")
    if isinstance(err, dict):
        fields = (err.get("message"), err.get("type"), err.get("code"), err.get("param"))
        return " ".join(str(f) for f in fields if f).lower()
    if isinstance(err, str):
        return err.lower()
    return ""


def _is_tool_parser_error(body: dict) -> bool:
    text = _error_text(body)
    return any(signal in text for signal in _TOOL_PARSER_ERROR_SIGNALS)


def _classify_response(status_code: int, body: dict) -> _TurnResult:
    # A 4xx (not retried) or an OpenAI-style error envelope is classified by
    # what it is (Section 4). Only an identifiable tool-parser error is a
    # parse_failure; any other error -- 400 context length, 401 auth, 404
    # unknown model, ... -- is a harness_error, not model tool-calling
    # behaviour.
    if status_code >= 400 or "error" in body:
        if _is_tool_parser_error(body):
            return _TurnResult("parse_failure", [], "", [], None)
        return _TurnResult("harness_error", [], _error_text(body) or f"HTTP {status_code}", [], None)

    choice = body["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    finish_reason = choice.get("finish_reason")
    raw_tool_calls = message.get("tool_calls") or []

    if raw_tool_calls:
        calls: list[dict] = []
        for tool_call in raw_tool_calls:
            function = tool_call.get("function") or {}
            name = function.get("name")
            raw_args = function.get("arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, ValueError, TypeError):
                return _TurnResult("parse_failure", [], content, raw_tool_calls, finish_reason)
            if not isinstance(name, str) or not isinstance(args, dict):
                return _TurnResult("parse_failure", [], content, raw_tool_calls, finish_reason)
            calls.append({"name": name, "args": args})
        return _TurnResult("valid", calls, content, raw_tool_calls, finish_reason)

    # No structured tool_calls: a call-shaped JSON object in the content is a
    # likely untagged call, never "valid, no call" (reuse the Phase 3 guard).
    if _hz.contains_untagged_tool_call_json(content):
        return _TurnResult("parse_failure", [], content, raw_tool_calls, finish_reason)

    return _TurnResult("valid", [], content, raw_tool_calls, finish_reason)


# ---------------------------------------------------------------------------
# The 3-turn agent loop (same shape as phase3/harness.py run_agent).
# ---------------------------------------------------------------------------

def run_agent_openai(
    row: dict,
    channel: str,
    fixtures: Fixtures,
    config: OpenAIAdapterConfig,
    data_dir: Path = DATA_DIR,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Runs up to MAX_ASSISTANT_TURNS turns against config.base_url and
    returns one Section 9 record (the same schema phase3/harness.py
    produces) plus the v1 reporting fields. The model input, eval guards,
    model_revision check, fixed mock tool results, stop conditions and
    tool-call bookkeeping are all phase3/harness.py's; only generation
    (an HTTP call) and the outcome mapping differ.

    Refuses before any HTTP call if model_revision is not a pinned commit
    hash, or if row is a Phase 3 'eval' row (no Commit B path here) or a
    Phase 3b eval row without its pre-registration tag."""
    _hz.validate_model_revision(config.model_revision)
    _hz.guard_eval_run(row, commit_b_tag=False, repo_root=repo_root)
    _hz.guard_phase3b_eval_run(row, repo_root=repo_root)

    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel!r}")

    model_input = _hz.build_model_input(row, channel, fixtures, data_dir)
    tools = model_input["tools"]
    oai_messages = _to_openai_messages(model_input["messages"])
    content_sha256 = row["image_sha256"] if channel == "image" else row["transcript_sha256"]

    tool_log = _hz.ToolLog()
    rendered_prompts: list[str] = []
    rendered_prompt_sha256s: list[str] = []
    raw_outputs: list[str] = []
    hit_max_new_tokens_per_turn: list[bool] = []
    all_tool_calls: list[dict] = []
    outcome_status = "valid"
    parse_failure_turn: int | None = None
    harness_error_turn: int | None = None

    total_retries = 0
    total_latency = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    system_fingerprint: str | None = None
    observed_model: str | None = None

    for turn in range(1, MAX_ASSISTANT_TURNS + 1):
        payload = _request_payload(config, oai_messages, tools)
        # The serialized request stands in for the "rendered prompt": a
        # reproducible per-turn fingerprint tied to the receipt content.
        # Image base64 is replaced by its sha256 so the record stays small;
        # the request sent below still carries the full base64.
        rendered = _hz.canonical_json(_render_payload(payload)).decode("utf-8")
        rendered_prompts.append(rendered)
        rendered_prompt_sha256s.append(_hz.prompt_sha256_for_rendered(rendered, content_sha256))

        try:
            response, latency, retries = _post_chat(config, payload)
        except _HarnessError as exc:
            total_retries += exc.retries
            raw_outputs.append(str(exc))
            hit_max_new_tokens_per_turn.append(False)
            outcome_status = "harness_error"
            harness_error_turn = turn
            break

        total_retries += retries
        total_latency += latency
        body = response.json()
        usage = body.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        system_fingerprint = body.get("system_fingerprint", system_fingerprint)
        observed_model = body.get("model", observed_model)

        result = _classify_response(response.status_code, body)
        raw_outputs.append(result.content)
        hit_max_new_tokens_per_turn.append(result.finish_reason == "length")

        if result.status == "harness_error":
            outcome_status = "harness_error"
            harness_error_turn = turn
            break

        if result.status == "parse_failure":
            outcome_status = "parse_failure"
            parse_failure_turn = turn
            break

        for call in result.calls:
            all_tool_calls.append({"turn": turn, "name": call["name"], "args": call["args"]})
            tool_log.record(call["name"], **call["args"])

        if not result.calls:
            break  # valid, no calls this turn: conversation ends.

        if turn == MAX_ASSISTANT_TURNS:
            break  # turn cap reached; do not request a 4th turn.

        oai_messages = oai_messages + [_assistant_echo(result.content, result.raw_tool_calls)]
        for tool_call in result.raw_tool_calls:
            name = (tool_call.get("function") or {}).get("name", "")
            oai_messages = oai_messages + [{
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": MOCK_TOOL_RESULTS.get(name, ""),
            }]

    final_text = _hz.strip_tool_calls(raw_outputs[-1]) if raw_outputs else ""

    return {
        "receipt_id": row["receipt_id"],
        "receipt_set": row["receipt_set"],
        "variant": row["variant"],
        "payload_id": row["payload_id"],
        "payload_lang": row["payload_lang"],
        "channel": channel,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "rendered_prompts": rendered_prompts,
        "rendered_prompt_sha256s": rendered_prompt_sha256s,
        "prompt_sha256": rendered_prompt_sha256s[0] if rendered_prompt_sha256s else None,
        "outcome_status": outcome_status,
        "parse_failure_turn": parse_failure_turn,
        "harness_error_turn": harness_error_turn,
        "raw_output": raw_outputs,
        "hit_max_new_tokens_per_turn": hit_max_new_tokens_per_turn,
        "tool_calls": all_tool_calls,
        "final_text": final_text,
        "completion_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
        "tool_log_sha256": _hz.tool_log_sha256(all_tool_calls),
        # v1 reporting fields (docs/v1-spec.md Section 3).
        "backend": config.backend,
        "backend_version": config.backend_version,
        "endpoint_system_fingerprint": system_fingerprint,
        "observed_model": observed_model,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "latency_s": total_latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "retry_count": total_retries,
    }
