#!/usr/bin/env python3
"""Phase 4 pre-training checks (docs/phase4-prereg.md section 9) AND the shared
assistant-token loss-mask function.

`assistant_token_labels` is the single source of the loss mask: it is imported
unchanged by phase4/train.py and the tests (do not reimplement it).

Run as a script: `python phase4/checks.py` (optionally --train-file / --config).
It prints each check and EXITS NON-ZERO if any check fails: a disjointness
overlap, Korean or an image in the training set, any gold tool-call argument
outside the loss span, or any example longer than max_seq_len (read from
phase4/train_config.yaml, not a recomputed maximum)."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DEFAULT_TRAIN = REPO / "phase4" / "data" / "train" / "train.jsonl"
DEFAULT_CONFIG = REPO / "phase4" / "train_config.yaml"


def assistant_token_labels(messages, tok, tools):
    """Returns (input_ids, labels, assistant_spans) for one chat example.

    Qwen's chat template has no {% generation %} tag, so return_assistant_tokens_mask
    is unusable. We compute the mask by cumulative prefixes: for each assistant
    message i, the loss span is [ len(render(messages[:i], add_generation_prompt=True)),
    len(render(messages[:i+1], add_generation_prompt=False)) ) -- from the end of the
    "<|im_start|>assistant\\n" header (excluded) through the content and its <|im_end|>.
    """
    def render(m, add_gen):
        return tok.apply_chat_template(m, tools=tools, add_generation_prompt=add_gen, tokenize=False)
    def ids(m, add_gen=False):
        return tok(render(m, add_gen), add_special_tokens=False)["input_ids"]
    full = ids(messages)
    labels = [-100] * len(full)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(ids(messages[:i], add_gen=True))
        prefix = ids(messages[:i + 1])
        end = len(prefix)
        assert full[:end] == prefix, f"prefix inconsistent at msg {i}"
        for j in range(start, end):
            labels[j] = full[j]
        spans.append((start, end))
    return full, labels, spans


def _norm(s):
    return " ".join(str(s).lower().split())


def run_checks(train_file: Path = DEFAULT_TRAIN, config_path: Path = DEFAULT_CONFIG, verbose: bool = True):
    """Runs all four section-9 checks. Returns the list of failure strings
    (empty == all pass). Reads max_seq_len from config_path."""
    import yaml
    sys.path.insert(0, str(REPO / "phase3"))
    import make_receipts as mr
    from transformers import AutoTokenizer
    import importlib.util

    def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    def say(*a):
        if verbose: print(*a)

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    max_seq_len = cfg["train"]["max_seq_len"]
    tools = json.loads((REPO / "phase3" / "fixtures" / "tools.json").read_text())

    train = jl(train_file)
    eval_gt = jl(REPO / "phase4" / "data" / "eval" / "receipts.jsonl")
    inj = json.loads((REPO / "phase4" / "data" / "injection_templates.json").read_text())["templates"]
    auth = json.loads((REPO / "phase4" / "data" / "authorised_email.json").read_text())
    phase3_vendors = mr.VENDORS
    spec = importlib.util.spec_from_file_location("p3b_mr", REPO / "phase3b" / "make_receipts.py")
    p3b = importlib.util.module_from_spec(spec); sys.modules["p3b_mr"] = p3b; spec.loader.exec_module(p3b)
    phase3b_pool = p3b.NEW_VENDOR_POOL
    phase3b_gt = jl(REPO / "phase3b" / "data" / "receipts.jsonl")
    phase3b_eval20 = {r["vendor"] for r in phase3b_gt if int(r["receipt_id"].split("_")[-1]) <= 20}
    phase4_eval_vendors = {r["vendor"] for r in eval_gt}

    failures = []

    # ---- CHECK 1: disjointness ----
    say("=" * 66); say("CHECK 1 - DISJOINTNESS"); say("=" * 66)
    train_vendors = set()
    for e in train:
        for m in e["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                v = m["tool_calls"][0]["function"]["arguments"].get("vendor")
                if v:
                    train_vendors.add(v)
    forbidden = {_norm(v) for v in list(phase3_vendors) + list(phase3b_pool) + list(phase4_eval_vendors)}
    ov = {_norm(v) for v in train_vendors} & forbidden
    say(f"training vendors {len(train_vendors)} | overlap with (phase3+phase3b+phase4_eval) {len(ov)} {sorted(ov)[:5]}")
    if ov:
        failures.append(f"vendor overlap: {sorted(ov)[:5]}")
    train_ids = {e["receipt_id"] for e in train}; eval_ids = {r["receipt_id"] for r in eval_gt}
    if train_ids & eval_ids:
        failures.append("train/eval receipt id overlap")
    p1 = {_norm("accounts-archive@example.com"), _norm("Receipt copy")}
    if {_norm(t["to"]) for t in inj} & p1 or {_norm(t["subject"]) for t in inj} & p1:
        failures.append("injection reuses P1 address/subject")
    ta = {_norm(x) for x in auth["train"]["addresses"]}; ea = {_norm(x) for x in auth["eval"]["addresses"]}
    tw = {_norm(x) for x in auth["train"]["wordings"]}; ew = {_norm(x) for x in auth["eval"]["wordings"]}
    if (ta & ea) or (tw & ew):
        failures.append("authorised-email train/eval overlap")
    if {_norm(v) for v in train_vendors} & {_norm(v) for v in phase3b_eval20}:
        failures.append("training vendor in phase3b_eval_01..20")
    kor = any(any('가' <= c <= '힣' for c in json.dumps(e, ensure_ascii=False)) for e in train)
    img = any("image" in json.dumps(e) for e in train)
    say(f"training has Korean: {kor} | has images: {img}")
    if kor:
        failures.append("Korean in training set")
    if img:
        failures.append("image in training set")

    # ---- CHECK 2+3+4: length (vs config max_seq_len), loss mask, truncation ----
    say("=" * 66); say(f"CHECK 2-4 - LENGTH (max_seq_len={max_seq_len} from config) + LOSS MASK"); say("=" * 66)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", revision=REV)
    lengths = []
    over = 0
    for e in train:
        full, labels, spans = assistant_token_labels(e["messages"], tok, tools)
        lengths.append(len(full))
        if len(full) > max_seq_len:
            over += 1
            failures.append(f"example {e.get('receipt_id')} length {len(full)} > max_seq_len {max_seq_len}")
        loss_text = tok.decode([t for t, l in zip(full, labels) if l != -100])
        for m in e["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for v in m["tool_calls"][0]["function"]["arguments"].values():
                    if str(v) not in loss_text:
                        failures.append(f"gold tool-call arg {v!r} outside loss span in {e.get('receipt_id')}")
    say(f"n={len(lengths)} | observed max={max(lengths)} | over max_seq_len={over}")

    say("=" * 66)
    say("RESULT:", "PASS" if not failures else f"FAIL ({len(failures)}): " + "; ".join(failures[:6]))
    return failures


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Phase 4 pre-training checks (exits non-zero on failure)")
    p.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    failures = run_checks(args.train_file, args.config)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
