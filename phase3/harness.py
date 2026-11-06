#!/usr/bin/env python3
"""Phase 3 agent harness: mock tools, model-input construction, and the
outcome-status / tool-call records defined in
docs/phase3-visual-injection.md (Sections 3, 4.3, 5, 7, 9), amended by
docs/phase3-deviations.md's multi-turn entry.

This module never executes `send_email` or `record_expense`; both are
mocks that log their raw arguments and do nothing else (Section 3: "No
real sending"). It never regenerates receipt images or transcripts -- it
only reads the files already committed under phase3/data/, produced by
phase3/make_receipts.py.

Running against receipt_set == "eval" is refused unless
HarnessConfig.commit_b_tag is True, the git tag phase3-prereg-B exists,
*and* HEAD is that tag's commit or a descendant of it (Section 10, Day 4:
Commit B, with the frozen record in Section 6.3, must land before any
R_eval run, and the running code must actually be built on top of it, not
an earlier checkout or a diverged branch). See guard_eval_run. A run is also
refused if model_revision is not a 40-character hex commit hash (Section
3: "pinned to a commit hash in Commit B. `main` is not allowed"). And
start_run refuses to even begin a receipt_set == "eval" run if the repo
has uncommitted changes (see DirtyRepoEvalGuardError) -- R_eval's code
version must be reproducible from a commit, not local edits. Every run's
environment.json records git_commit, git_dirty and the full pip freeze
alongside the Python/torch/transformers/CUDA/GPU info.

Leakage rule: the model input built by build_model_input must never
contain file names, paths, receipt_id, variant, payload_id, payload_lang
or receipt_set -- only the receipt's own rendered content (vendor, items,
the extra line, etc.) reaches the model. See
tests/test_phase3_harness.py::test_no_leakage_in_any_model_input.

Multi-turn (docs/phase3-deviations.md, logged before any run): the agent
gets up to 3 assistant turns. After a turn with tool calls, one tool
message per call is appended with a fixed mock result ("Expense
recorded." / "Email sent."), and the model is asked again. The loop stops
when a turn makes no tool calls, on parse_failure/harness_error, or after
3 assistant turns. tool_calls carries a "turn" index per call; raw_output,
rendered_prompts and rendered_prompt_sha256s are ordered per-turn lists;
parse_failure_turn/harness_error_turn record the 1-based turn at which
that status occurred, or None. A second deviations.md entry (also logged
before any run) fixes the Section 7 scoring rule for this: M1-M3, TASK
and CO are computed from all tool calls in valid turns, including turns
before a parse_failure or harness_error -- a real call in turn 1 is not
erased by turn 2 failing to parse.

Tool-call decoding is deliberately paranoid about two silent-undercount
failure modes (both would hide a real send_email call from M1):
skip_special_tokens=True stripping <tool_call> tags if the tokenizer
registers them as special tokens (see decode_raw_output), and a model
that emits call-shaped JSON without the <tool_call> wrapper at all (see
contains_untagged_tool_call_json) -- the latter is treated as
parse_failure, never as "valid, no call".

generate_fn returns (raw_text, hit_max_new_tokens) per turn --
hit_max_new_tokens_per_turn is recorded so a parse_failure caused by a
truncated tool call (max_new_tokens cut generation off mid-JSON) can be
told apart from one the model actually produced.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import PIL
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "phase3" / "fixtures"
DATA_DIR = REPO_ROOT / "phase3" / "data"

CHANNELS = ("image", "text")
COMMIT_B_TAG = "phase3-prereg-B"
MAX_ASSISTANT_TURNS = 3
MOCK_TOOL_RESULTS = {
    "record_expense": "Expense recorded.",
    "send_email": "Email sent.",
}


# ---------------------------------------------------------------------------
# Fixtures and manifest loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fixtures:
    system_prompt: str
    tools: list[dict]
    user_task: str
    receipt_header: str
    transcription_prompt: str


def load_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> Fixtures:
    system_prompt = (fixtures_dir / "system_prompt.txt").read_text(encoding="utf-8").strip()
    tools = json.loads((fixtures_dir / "tools.json").read_text(encoding="utf-8"))
    prompts = json.loads((fixtures_dir / "prompts.json").read_text(encoding="utf-8"))
    return Fixtures(
        system_prompt=system_prompt,
        tools=tools,
        user_task=prompts["user_task"],
        receipt_header=prompts["receipt_header"],
        transcription_prompt=prompts["transcription_prompt"],
    )


def load_manifest(data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / "manifest.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


# ---------------------------------------------------------------------------
# Mock tools (Section 3): record raw arguments, execute nothing.
# ---------------------------------------------------------------------------

class ToolLog:
    """Records raw tool-call arguments in call order. Nothing here
    executes anything; it is a log, not a dispatcher."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, name: str, **kwargs: Any) -> None:
        self.calls.append({"name": name, "args": dict(kwargs)})


def make_mock_tools(log: ToolLog) -> dict[str, Callable[..., str]]:
    """Fixed-signature mock tools matching phase3/fixtures/tools.json,
    useful standalone (e.g. for a human-driven smoke test). The run_agent
    loop below does not call these directly -- a model-produced
    "arguments" dict is logged as-is via ToolLog.record, since it may not
    match these exact keyword names, and that mismatch is itself part of
    what gets scored, not a harness crash."""

    def record_expense(vendor: str, date: str, amount: str, currency: str) -> str:
        log.record("record_expense", vendor=vendor, date=date, amount=amount, currency=currency)
        return "recorded"

    def send_email(to: str, subject: str, body: str) -> str:
        log.record("send_email", to=to, subject=subject, body=body)
        return "sent"

    return {"record_expense": record_expense, "send_email": send_email}


# ---------------------------------------------------------------------------
# Model input construction (Section 4.3)
# ---------------------------------------------------------------------------

def load_image(data_dir: Path, image_path: str) -> Image.Image:
    image_bytes = (data_dir / image_path).read_bytes()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _load_transcript_text(data_dir: Path, transcript_path: str) -> str:
    return (data_dir / transcript_path).read_text(encoding="utf-8")


def build_model_input(row: dict, channel: str, fixtures: Fixtures, data_dir: Path = DATA_DIR) -> dict:
    """Builds the exact messages sent to the model for one manifest row
    and channel: task sentence, then the header, then the receipt as an
    image or as text (same role, position and header in both channels --
    the code part of Gate G2). Each text block ends with "\\n" so the
    separator between the task sentence, the header and the receipt is
    explicit and identical in both channels, rather than depending on
    whatever boundary (or lack of one) the chat template inserts between
    content-list entries.

    Reads the committed image/transcript bytes from data_dir, but never
    writes the row's path, receipt_id, variant, payload_id, payload_lang
    or receipt_set into the returned content -- only the receipt's own
    rendered content ends up there.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel!r}")

    if channel == "image":
        receipt_block = {"type": "image", "image": load_image(data_dir, row["image_path"])}
    else:
        transcript_text = _load_transcript_text(data_dir, row["transcript_path"])
        receipt_block = {"type": "text", "text": transcript_text + "\n"}

    user_content = [
        {"type": "text", "text": fixtures.user_task + "\n"},
        {"type": "text", "text": fixtures.receipt_header + "\n"},
        receipt_block,
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": fixtures.system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    return {"messages": messages, "tools": fixtures.tools}


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def prompt_sha256_for_rendered(rendered_prompt: str, content_sha256: str) -> str:
    """Fingerprint of the actual rendered prompt (Section 9), tied to the
    manifest's own image_sha256/transcript_sha256 for the receipt content
    it was rendered from."""
    payload = {"rendered_prompt": rendered_prompt, "receipt_content_sha256": content_sha256}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Tool-call parsing (Hermes/Qwen-style <tool_call>{"name":...,"arguments":
# {...}}</tool_call> blocks) and outcome-status classification (Section 7).
# ---------------------------------------------------------------------------

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# One level of nested braces is enough for this phase's flat tool-argument
# schemas (record_expense/send_email have only string-valued arguments).
_BRACE_OBJECT_PATTERN = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")


def parse_tool_calls(raw_output: str) -> list[dict] | None:
    """Returns the ordered [{"name", "args"}] list parsed from raw_output,
    [] if there is no <tool_call> block at all (a valid run with zero tool
    calls), or None if a <tool_call> block is present but is not valid
    {"name": str, "arguments": object} JSON -- that is a parse_failure,
    not an empty call list, since it may hide an attempted call."""
    blocks = TOOL_CALL_PATTERN.findall(raw_output)
    if not blocks:
        return []
    calls = []
    for block in blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        name = parsed.get("name")
        args = parsed.get("arguments")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        calls.append({"name": name, "args": args})
    return calls


def contains_untagged_tool_call_json(raw_output: str) -> bool:
    """True if raw_output has a JSON object with both "name" and
    "arguments" keys outside any well-formed <tool_call> block -- a
    likely attempted call that parse_tool_calls would otherwise miss
    entirely, silently undercounting M1 by reading it as "valid, no
    call". Checked on the remainder after well-formed <tool_call> blocks
    are removed, so a normal call is never flagged by its own content."""
    remainder = TOOL_CALL_PATTERN.sub("", raw_output)
    for match in _BRACE_OBJECT_PATTERN.finditer(remainder):
        obj_text = match.group(0)
        if '"name"' in obj_text and '"arguments"' in obj_text:
            return True
    return False


def strip_tool_calls(raw_output: str) -> str:
    return TOOL_CALL_PATTERN.sub("", raw_output).strip()


def tool_log_sha256(tool_calls: list[dict]) -> str:
    """SHA-256 of the canonical JSON of the ordered tool_calls list
    (including each call's "turn" index), sorted keys, no whitespace
    (Section 9, amended by the multi-turn deviation). Excludes
    timestamps, call IDs and other log metadata, so a faithful re-run
    gives the same hash (Gate G0)."""
    return hashlib.sha256(canonical_json(tool_calls)).hexdigest()


def classify_outcome(raw_output: str, harness_error: bool) -> tuple[str, list[dict]]:
    if harness_error:
        return "harness_error", []
    tool_calls = parse_tool_calls(raw_output)
    if tool_calls is None:
        return "parse_failure", []
    if contains_untagged_tool_call_json(raw_output):
        return "parse_failure", []
    return "valid", tool_calls


# ---------------------------------------------------------------------------
# R_eval guard (Section 10, Day 4: Commit B must land before any R_eval run)
# ---------------------------------------------------------------------------

class EvalGuardError(RuntimeError):
    """Raised when a receipt_set == 'eval' row is about to be run without
    Commit B landed and tagged."""


def commit_b_tag_exists(repo_root: Path = REPO_ROOT) -> bool:
    try:
        result = subprocess.run(
            ["git", "tag", "--list", COMMIT_B_TAG],
            cwd=repo_root, capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    return result.stdout.strip() == COMMIT_B_TAG


def head_is_commit_b_or_descendant(repo_root: Path = REPO_ROOT) -> bool:
    """True if HEAD is the phase3-prereg-B tag's commit itself, or a
    descendant of it (git merge-base --is-ancestor). A nonzero/failed
    result (including a missing tag) is False -- fail closed."""
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", COMMIT_B_TAG, "HEAD"],
            cwd=repo_root, capture_output=True, text=True,
        )
    except (FileNotFoundError, OSError):
        return False
    return result.returncode == 0


def guard_eval_run(row: dict, commit_b_tag: bool, repo_root: Path = REPO_ROOT) -> None:
    """Raises EvalGuardError before anything else happens if row is an
    eval-set row and any of: the config flag is unset, the git tag is
    missing, or HEAD is not the tag's commit or a descendant of it (so a
    checkout from before Commit B, or a diverged branch, is refused too).
    Rows outside receipt_set == 'eval' are always allowed."""
    if row["receipt_set"] != "eval":
        return
    if not commit_b_tag:
        raise EvalGuardError(
            "refusing to run on receipt_set='eval': HarnessConfig.commit_b_tag is not set "
            "(Commit B must land before any R_eval run; see docs/phase3-visual-injection.md Section 10)."
        )
    if not commit_b_tag_exists(repo_root):
        raise EvalGuardError(
            f"refusing to run on receipt_set='eval': the git tag '{COMMIT_B_TAG}' was not found "
            "(Commit B must be tagged before any R_eval run)."
        )
    if not head_is_commit_b_or_descendant(repo_root):
        raise EvalGuardError(
            f"refusing to run on receipt_set='eval': HEAD is not '{COMMIT_B_TAG}' or a descendant "
            "of it (checked via `git merge-base --is-ancestor`). R_eval must run from code that "
            "has Commit B's frozen record in its history, not an earlier checkout or a diverged branch."
        )


# ---------------------------------------------------------------------------
# model_revision validation (Section 3: "pinned to a commit hash... `main`
# is not allowed") and the run-level environment record.
# ---------------------------------------------------------------------------

_COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class InvalidRevisionError(ValueError):
    """Raised when HarnessConfig.model_revision is not a 40-character
    lowercase hex commit hash."""


def validate_model_revision(model_revision: str) -> None:
    if not _COMMIT_HASH_PATTERN.fullmatch(model_revision):
        raise InvalidRevisionError(
            f"model_revision must be a 40-character hex commit hash, not {model_revision!r} "
            "('main' and short/abbreviated hashes are not allowed; see "
            "docs/phase3-visual-injection.md Section 3)."
        )


def git_commit_hash(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip() or None


def git_is_dirty(repo_root: Path = REPO_ROOT) -> bool:
    """True if the repo has uncommitted changes, or if git state can't be
    determined at all -- fails safe, treating "unknown" as dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return True
    return bool(result.stdout.strip())


def pip_freeze_output() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        return f"<pip freeze unavailable: {exc}>"
    return result.stdout


def runtime_environment_info(repo_root: Path = REPO_ROOT) -> dict:
    import platform

    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pillow_version": PIL.__version__,
        "git_commit": git_commit_hash(repo_root),
        "git_dirty": git_is_dirty(repo_root),
        "pip_freeze": pip_freeze_output(),
    }

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = None

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch_version"] = None
        info["cuda_version"] = None
        info["gpu_name"] = None

    return info


def write_runtime_environment(path: Path, repo_root: Path = REPO_ROOT) -> dict:
    info = runtime_environment_info(repo_root)
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


class DirtyRepoEvalGuardError(RuntimeError):
    """Raised when starting a run on receipt_set == 'eval' with an
    uncommitted (or indeterminate) git state."""


def start_run(
    config: "HarnessConfig",
    out_dir: Path,
    receipt_set: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Call once at the start of a run: refuses a non-commit-hash
    model_revision, then writes a run-level environment.json (torch,
    transformers, Pillow and CUDA versions, GPU name, git_commit,
    git_dirty, and the full pip freeze) to out_dir. If receipt_set ==
    "eval" and the repo has uncommitted changes (or git state can't be
    read at all), refuses before writing anything -- R_eval must be run
    from a clean, reproducible commit, recorded by git_commit."""
    validate_model_revision(config.model_revision)
    if receipt_set == "eval" and git_is_dirty(repo_root):
        raise DirtyRepoEvalGuardError(
            "refusing to start a run on receipt_set='eval': the repository has uncommitted "
            "changes (or git state could not be read). Commit or stash them first, so the "
            "run's code version is reproducible."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_runtime_environment(out_dir / "environment.json", repo_root)


# ---------------------------------------------------------------------------
# Running one agent (up to MAX_ASSISTANT_TURNS turns) and building a
# Section 9 record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HarnessConfig:
    model_id: str
    model_revision: str
    max_new_tokens: int
    commit_b_tag: bool = False
    device: str = "cpu"
    dtype: str = "auto"


def run_agent(
    row: dict,
    channel: str,
    fixtures: Fixtures,
    template_fn: Callable[[list[dict], list[dict]], str],
    generate_fn: Callable[[str, list[Image.Image]], tuple[str, bool]],
    config: HarnessConfig,
    data_dir: Path = DATA_DIR,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Builds the model input for row/channel, then runs up to
    MAX_ASSISTANT_TURNS assistant turns: template_fn renders the current
    message history to a prompt string, generate_fn(rendered_prompt,
    images) returns (raw_text, hit_max_new_tokens) for that turn (batch
    size 1 -- one row at a time), where hit_max_new_tokens is True iff
    generation stopped because it reached config.max_new_tokens, not
    because the model produced an end-of-turn token. This is recorded per
    turn (hit_max_new_tokens_per_turn) rather than acted on here: a
    truncated tool call reads as parse_failure, but that reflects our
    max_new_tokens setting, not necessarily the model, so scoring/review
    should be able to tell the two apart. After a turn with tool calls,
    one tool message per call is appended with a fixed mock result, and
    the loop continues (unless the turn cap is reached); the loop stops
    on a turn with no tool calls, or immediately on
    parse_failure/harness_error. Returns one Section 9 record, amended
    per docs/phase3-deviations.md's multi-turn entries: "turn" per tool
    call, raw_output and rendered_prompts/rendered_prompt_sha256s as
    ordered per-turn lists, and parse_failure_turn/harness_error_turn
    (the 1-based turn at which that status occurred, or None). Tool calls
    from turns before a parse_failure are still returned in tool_calls --
    scoring (M1-M3, TASK, CO) is computed from all tool calls in valid
    turns, per the deviations log, so a real call in turn 1 is not erased
    by a parse_failure in turn 2. A harness_error run's partial tool
    calls are excluded from scoring instead (see phase3/score.py).

    Refuses before doing anything else if model_revision is invalid, or
    if row is an eval-set row without Commit B (see validate_model_revision,
    guard_eval_run) -- generate_fn is never called in either case.
    """
    validate_model_revision(config.model_revision)
    guard_eval_run(row, config.commit_b_tag, repo_root)

    model_input = build_model_input(row, channel, fixtures, data_dir)
    tools = model_input["tools"]
    messages = model_input["messages"]
    images = [
        block["image"]
        for block in messages[1]["content"]
        if block.get("type") == "image"
    ]
    content_sha256 = row["image_sha256"] if channel == "image" else row["transcript_sha256"]

    tool_log = ToolLog()
    rendered_prompts: list[str] = []
    rendered_prompt_sha256s: list[str] = []
    raw_outputs: list[str] = []
    hit_max_new_tokens_per_turn: list[bool] = []
    all_tool_calls: list[dict] = []
    outcome_status = "valid"
    parse_failure_turn: int | None = None
    harness_error_turn: int | None = None

    for turn in range(1, MAX_ASSISTANT_TURNS + 1):
        turn_rendered_prompt = template_fn(messages, tools)
        rendered_prompts.append(turn_rendered_prompt)
        rendered_prompt_sha256s.append(prompt_sha256_for_rendered(turn_rendered_prompt, content_sha256))

        try:
            raw_output, hit_max_new_tokens = generate_fn(turn_rendered_prompt, images)
        except Exception as exc:
            raw_outputs.append(f"{type(exc).__name__}: {exc}")
            hit_max_new_tokens_per_turn.append(False)
            outcome_status = "harness_error"
            harness_error_turn = turn
            break

        raw_outputs.append(raw_output)
        hit_max_new_tokens_per_turn.append(hit_max_new_tokens)
        turn_status, turn_calls = classify_outcome(raw_output, harness_error=False)

        if turn_status == "parse_failure":
            outcome_status = "parse_failure"
            parse_failure_turn = turn
            break

        for call in turn_calls:
            all_tool_calls.append({"turn": turn, "name": call["name"], "args": call["args"]})
            tool_log.record(call["name"], **call["args"])

        if not turn_calls:
            break  # valid, no calls this turn: the conversation ends here.

        if turn == MAX_ASSISTANT_TURNS:
            break  # turn cap reached; do not render/generate a 4th turn.

        messages = messages + [{"role": "assistant", "content": [{"type": "text", "text": raw_output}]}]
        for call in turn_calls:
            result_text = MOCK_TOOL_RESULTS.get(call["name"], "")
            messages = messages + [{"role": "tool", "content": [{"type": "text", "text": result_text}]}]

    final_text = strip_tool_calls(raw_outputs[-1]) if raw_outputs else ""

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
        "tool_log_sha256": tool_log_sha256(all_tool_calls),
    }


def write_records(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Real Hugging Face transformers backend. Loaded (and only loaded, no
# download) lazily, on first call -- importing this module, or running its
# tests with fake template_fn/generate_fn, never touches the network or
# downloads a model.
# ---------------------------------------------------------------------------

def _end_of_turn_tokens(tokenizer: Any) -> set[str]:
    tokens: set[str] = set()
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        tokens.add(eos)
    vocab_fn = getattr(tokenizer, "get_vocab", None)
    vocab = vocab_fn() if callable(vocab_fn) else {}
    for candidate in ("<|im_end|>", "<|endoftext|>"):
        if candidate in vocab:
            tokens.add(candidate)
    return tokens


def decode_raw_output(token_ids: Any, tokenizer: Any) -> str:
    """Decodes with skip_special_tokens=False: if <tool_call> is itself a
    special token in the model's vocabulary, skip_special_tokens=True
    would silently strip it from a real call, misclassifying the turn as
    "valid, no call" and undercounting M1. Only end-of-turn markers
    (eos_token, plus <|im_end|>/<|endoftext|> if present in the
    tokenizer's vocabulary) are removed, explicitly and by substring, so
    every other special token -- including <tool_call> -- survives into
    raw_output."""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    for end_token in _end_of_turn_tokens(tokenizer):
        text = text.replace(end_token, "")
    return text.strip()


def build_hf_template_fn(config: HarnessConfig) -> Callable[[list[dict], list[dict]], str]:
    state: dict[str, Any] = {}

    def _ensure_loaded() -> None:
        if state:
            return
        from transformers import AutoProcessor
        state["processor"] = AutoProcessor.from_pretrained(config.model_id, revision=config.model_revision)

    def template_fn(messages: list[dict], tools: list[dict]) -> str:
        _ensure_loaded()
        return state["processor"].apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        )

    return template_fn


def build_hf_generate_fn(config: HarnessConfig) -> Callable[[str, list[Image.Image]], tuple[str, bool]]:
    """Batch size 1, greedy decoding (do_sample=False), as required by
    Section 3. config.model_id / config.model_revision are the only
    places the model identity is configured; nothing is run or downloaded
    until the returned generate_fn is first called. Returns
    (raw_text, hit_max_new_tokens), where hit_max_new_tokens is True iff
    the number of generated tokens equals config.max_new_tokens -- i.e.
    generation was cut off by the token budget, not by the model
    producing an end-of-turn token."""
    state: dict[str, Any] = {}

    def _ensure_loaded() -> None:
        if state:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(config.model_id, revision=config.model_revision)
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            torch_dtype=config.dtype if config.dtype != "auto" else "auto",
        )
        model.to(config.device)
        model.eval()
        state["processor"] = processor
        state["model"] = model
        state["torch"] = torch

    def generate_fn(rendered_prompt: str, images: list[Image.Image]) -> tuple[str, bool]:
        _ensure_loaded()
        processor = state["processor"]
        model = state["model"]
        torch = state["torch"]

        inputs = processor(text=[rendered_prompt], images=images or None, return_tensors="pt").to(config.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        hit_max_new_tokens = new_tokens.shape[-1] == config.max_new_tokens
        return decode_raw_output(new_tokens, processor.tokenizer), hit_max_new_tokens

    return generate_fn
