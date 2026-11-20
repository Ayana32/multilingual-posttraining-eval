#!/usr/bin/env python3
"""Phase 4 pre-training checks (docs/phase4-prereg.md section 9) AND the shared
assistant-token loss-mask function.

`assistant_token_labels` is the single source of the loss mask: it is imported
unchanged by phase4/train.py and by the tests (do not reimplement it). The
four section-9 checks run only when this file is executed as a script.

Read-only. Run the checks with a venv that has transformers 5.22.2 (tokenizer
only, no weights)."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def assistant_token_labels(messages, tok, tools):
    """Returns (input_ids, labels, assistant_spans) for one chat example.

    Qwen's chat template has no {% generation %} tag, so return_assistant_tokens_mask
    is unusable. We compute the mask by cumulative prefixes: for each assistant
    message i, the loss span is [ len(render(messages[:i], add_generation_prompt=True)),
    len(render(messages[:i+1], add_generation_prompt=False)) ) -- i.e. from the end
    of the "<|im_start|>assistant\\n" header (excluded from loss) through the message's
    content and its <|im_end|>. All non-assistant tokens stay -100.
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


# --------------------------------------------------------------------------
# Section-9 checks (script only)
# --------------------------------------------------------------------------
def main():
    sys.path.insert(0, str(REPO / "phase3"))
    import make_receipts as mr
    from transformers import AutoTokenizer
    import importlib.util

    P4 = REPO / "phase4" / "data"
    TOOLS = json.loads((REPO / "phase3" / "fixtures" / "tools.json").read_text())

    def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    def norm(s): return " ".join(str(s).lower().split())

    train = jl(P4 / "train" / "train.jsonl")
    eval_gt = jl(P4 / "eval" / "receipts.jsonl")
    inj = json.loads((P4 / "injection_templates.json").read_text())["templates"]
    auth = json.loads((P4 / "authorised_email.json").read_text())

    phase3_vendors = mr.VENDORS
    spec = importlib.util.spec_from_file_location("p3b_mr", REPO / "phase3b" / "make_receipts.py")
    p3b = importlib.util.module_from_spec(spec); sys.modules["p3b_mr"] = p3b; spec.loader.exec_module(p3b)
    phase3b_pool = p3b.NEW_VENDOR_POOL
    phase3b_gt = jl(REPO / "phase3b" / "data" / "receipts.jsonl")
    phase3b_eval20 = {r["vendor"] for r in phase3b_gt if int(r["receipt_id"].split("_")[-1]) <= 20}
    phase4_eval_vendors = {r["vendor"] for r in eval_gt}
    train_vendors_used = set()
    for e in train:
        for m in e["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                a = m["tool_calls"][0]["function"]["arguments"]
                if a.get("vendor"): train_vendors_used.add(a["vendor"])

    print("=" * 70); print("CHECK 1 - DISJOINTNESS (case/whitespace-insensitive)"); print("=" * 70)
    fails = []
    forbidden_v = {norm(v) for v in list(phase3_vendors) + list(phase3b_pool) + list(phase4_eval_vendors)}
    tv = {norm(v) for v in train_vendors_used}
    ov = tv & forbidden_v
    print(f"training vendors: {len(train_vendors_used)} | forbidden: {len(forbidden_v)} | overlap: {len(ov)} {sorted(ov)[:5]}")
    if ov: fails.append("vendor overlap")
    p4e = {norm(v) for v in phase4_eval_vendors}
    print(f"phase4_eval vendors: {len(phase4_eval_vendors)} | from phase3 VENDORS: {len(p4e & {norm(v) for v in phase3_vendors})} | from phase3b pool: {len(p4e & {norm(v) for v in phase3b_pool})}")
    train_ids = {e['receipt_id'] for e in train}; eval_ids = {r['receipt_id'] for r in eval_gt}
    print(f"train vs eval receipt ids overlap: {len(train_ids & eval_ids)}")
    if train_ids & eval_ids: fails.append("receipt id overlap")
    p1 = {norm("accounts-archive@example.com"), norm("Receipt copy")}
    ia = {norm(t['to']) for t in inj}; isj = {norm(t['subject']) for t in inj}
    print(f"injection addr∩P1: {len(ia & p1)} | injection subj∩P1: {len(isj & p1)}")
    if (ia & p1) or (isj & p1): fails.append("injection reuses P1")
    ta = {norm(x) for x in auth['train']['addresses']}; ea = {norm(x) for x in auth['eval']['addresses']}
    tw = {norm(x) for x in auth['train']['wordings']}; ew = {norm(x) for x in auth['eval']['wordings']}
    ts = {norm(x) for x in auth['train']['subjects']}; es = {norm(x) for x in auth['eval']['subjects']}
    print(f"auth addr train∩eval: {len(ta & ea)} | wording: {len(tw & ew)} | subject: {len(ts & es)}")
    if (ta & ea) or (tw & ew) or (ts & es): fails.append("auth train/eval overlap")
    print(f"training vendors ∩ phase3b_eval_01..20: {len(tv & {norm(v) for v in phase3b_eval20})}")
    kor = any(any('가' <= c <= '힣' for c in json.dumps(e, ensure_ascii=False)) for e in train)
    img = any('image' in json.dumps(e) for e in train)
    print(f"training has Korean: {kor} | training has images: {img}")
    if kor or img: fails.append("korean/images in training")
    print("RESULT:", "PASS" if not fails else f"FAIL {fails}")

    print("\n" + "=" * 70); print("CHECK 2 - LENGTH (Qwen3-VL-4B tokenizer + chat template)"); print("=" * 70)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", revision=REV)
    lengths = sorted(len(assistant_token_labels(e["messages"], tok, TOOLS)[0]) for e in train)
    MAXLEN = lengths[-1]
    print(f"n={len(lengths)} | min={lengths[0]} | mean={sum(lengths)//len(lengths)} | p99={lengths[int(0.99*len(lengths))-1]} | MAX={MAXLEN}")
    print(f"MAX SEQUENCE LENGTH (number): {MAXLEN}")

    print("\n" + "=" * 70); print("CHECK 3 - LOSS MASK (one per type; gold tool-call args unmasked)"); print("=" * 70)
    seen = {}
    for e in train:
        seen.setdefault(e["type"], e)
    for typ in ("injected", "clean", "authorised_email"):
        e = seen[typ]; full, labels, spans = assistant_token_labels(e["messages"], tok, TOOLS)
        n_loss = sum(1 for x in labels if x != -100)
        loss_text = tok.decode([t for t, l in zip(full, labels) if l != -100])
        golds = []
        for m in e["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                golds += [str(v) for v in m["tool_calls"][0]["function"]["arguments"].values()]
        missing = [g for g in golds if g not in loss_text]
        print(f"\n[{typ}] {e['receipt_id']} tokens={len(full)} loss_tokens={n_loss} spans={spans}")
        print(f"  gold tool-call args all unmasked: {not missing} (missing {missing})")
        print(f"  loss region (first 180): {loss_text[:180]!r}")

    print("\n" + "=" * 70); print("CHECK 4 - NO TRUNCATION"); print("=" * 70)
    print(f"max_seq_len={MAXLEN}; examples exceeding it: {sum(1 for L in lengths if L > MAXLEN)} (longest == max_seq_len)")
    print(f"\nMAX_SEQ_LEN={MAXLEN}")


if __name__ == "__main__":
    main()
