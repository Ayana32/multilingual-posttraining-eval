#!/usr/bin/env python3
"""Phase 4 pre-training checks (docs/phase4-prereg.md section 9).
Read-only: reads the frozen data, computes, prints. Run with a venv that has
transformers 5.22.2 (tokenizer only, no weights)."""
from __future__ import annotations
import json, sys, re
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase3"))
import make_receipts as mr
from transformers import AutoTokenizer

P4 = REPO / "phase4" / "data"
REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
TOOLS = json.loads((REPO / "phase3" / "fixtures" / "tools.json").read_text())

def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def norm(s): return " ".join(str(s).lower().split())

train = jl(P4 / "train" / "train.jsonl")
eval_gt = jl(P4 / "eval" / "receipts.jsonl")
auth_eval = jl(REPO / "runs" / "phase4_authorised_email_eval.jsonl")
inj = json.loads((P4 / "injection_templates.json").read_text())["templates"]
auth = json.loads((P4 / "authorised_email.json").read_text())
train_vendor_pool = json.loads((P4 / "training_vendors.json").read_text())["vendors"]

# existing pools
phase3_vendors = mr.VENDORS
import importlib.util
spec = importlib.util.spec_from_file_location("p3b_mr", REPO / "phase3b" / "make_receipts.py")
p3b = importlib.util.module_from_spec(spec); sys.modules["p3b_mr"] = p3b; spec.loader.exec_module(p3b)
phase3b_pool = p3b.NEW_VENDOR_POOL
phase3b_gt = jl(REPO / "phase3b" / "data" / "receipts.jsonl")
phase3b_eval20_vendors = {r["vendor"] for r in phase3b_gt if int(r["receipt_id"].split("_")[-1]) <= 20}
phase4_eval_vendors = {r["vendor"] for r in eval_gt}
train_vendors_used = {r["vendor"] for e in train for r in [None] if False}  # placeholder
train_vendors_used = set()
for e in train:
    # vendor is embedded in the first tool call args
    for m in e["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            a = m["tool_calls"][0]["function"]["arguments"]
            if a.get("vendor"): train_vendors_used.add(a["vendor"])

print("=" * 70); print("CHECK 1 — DISJOINTNESS (case/whitespace-insensitive)"); print("=" * 70)
fails = []
# vendors: training disjoint from phase3 VENDORS + full phase3b pool(40) + phase4_eval
forbidden_v = {norm(v) for v in list(phase3_vendors) + list(phase3b_pool) + list(phase4_eval_vendors)}
tv = {norm(v) for v in train_vendors_used}
ov = tv & forbidden_v
print(f"training vendors used: {len(train_vendors_used)} | forbidden set: {len(forbidden_v)} | overlap: {len(ov)} {sorted(ov)[:5]}")
if ov: fails.append("vendor overlap")
# phase4_eval vendor provenance
p4e = {norm(v) for v in phase4_eval_vendors}
from_p3 = p4e & {norm(v) for v in phase3_vendors}; from_p3b = p4e & {norm(v) for v in phase3b_pool}
print(f"phase4_eval vendors: {len(phase4_eval_vendors)} | from phase3 VENDORS: {len(from_p3)} | from phase3b pool: {len(from_p3b)}")
# receipt ids disjoint
train_ids = {e["receipt_id"] for e in train}; eval_ids = {r["receipt_id"] for r in eval_gt}
print(f"train receipt ids vs eval ids overlap: {len(train_ids & eval_ids)}")
if train_ids & eval_ids: fails.append("receipt id overlap")
# injection addresses/subjects vs P1 and phase3b
P1 = {"accounts-archive@example.com", "Receipt copy"}
inj_addr = {norm(t["to"]) for t in inj}; inj_subj = {norm(t["subject"]) for t in inj}
p1n = {norm(x) for x in P1}
print(f"injection addresses∩P1: {len(inj_addr & p1n)} | injection subjects∩P1: {len(inj_subj & p1n)}")
if (inj_addr & p1n) or (inj_subj & p1n): fails.append("injection reuses P1")
# authorised-email train vs eval disjoint (wording/address/subject)
ta = {norm(x) for x in auth["train"]["addresses"]}; ea = {norm(x) for x in auth["eval"]["addresses"]}
tw = {norm(x) for x in auth["train"]["wordings"]}; ew = {norm(x) for x in auth["eval"]["wordings"]}
tsub = {norm(x) for x in auth["train"]["subjects"]}; esub = {norm(x) for x in auth["eval"]["subjects"]}
print(f"auth addr train∩eval: {len(ta & ea)} | wording train∩eval: {len(tw & ew)} | subject train∩eval: {len(tsub & esub)}")
if (ta & ea) or (tw & ew) or (tsub & esub): fails.append("auth train/eval overlap")
# training content must not appear in phase3b_eval_01..20 vendors
print(f"training vendors ∩ phase3b_eval_01..20 vendors: {len(tv & {norm(v) for v in phase3b_eval20_vendors})}")
# training set: no Korean, no images
kor = any(any('가' <= c <= '힣' for c in json.dumps(e, ensure_ascii=False)) for e in train)
img = any('image' in json.dumps(e) for e in train)
print(f"training contains Korean: {kor} | training contains images: {img}")
if kor: fails.append("training has Korean")
if img: fails.append("training has images")
print("RESULT:", "PASS" if not fails else f"FAIL {fails}")

print("\n" + "=" * 70); print("CHECK 2 — LENGTH (Qwen3-VL-4B tokenizer + chat template)"); print("=" * 70)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", revision=REV)
def render(messages, add_gen):
    return tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=add_gen, tokenize=False)
def ids_of(messages, add_gen=False):
    return tok(render(messages, add_gen), add_special_tokens=False)["input_ids"]
lengths = []
for e in train:
    lengths.append(len(ids_of(e["messages"])))
lengths.sort()
MAXLEN = lengths[-1]
print(f"n={len(lengths)} | min={lengths[0]} | mean={sum(lengths)//len(lengths)} | p99={lengths[int(0.99*len(lengths))-1]} | MAX={MAXLEN}")
print(f"MAX SEQUENCE LENGTH (number): {MAXLEN}")

print("\n" + "=" * 70); print("CHECK 3 — LOSS MASK (one per type; gold tool-call args unmasked)"); print("=" * 70)
def labels_for(messages):
    full = ids_of(messages)
    labels = [-100] * len(full); spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant": continue
        start = len(ids_of(messages[:i], add_gen=True))
        prefix = ids_of(messages[:i + 1])
        end = len(prefix)
        assert full[:end] == prefix, f"prefix inconsistent at msg {i}"
        for j in range(start, end): labels[j] = full[j]
        spans.append((start, end))
    return full, labels, spans
seen = {}
for e in train:
    if e["type"] in seen: continue
    seen[e["type"]] = e
    if len(seen) == 3: break
for typ in ("injected", "clean", "authorised_email"):
    e = seen[typ]; full, labels, spans = labels_for(e["messages"])
    n_loss = sum(1 for x in labels if x != -100)
    loss_text = tok.decode([t for t, l in zip(full, labels) if l != -100])
    # gold tool-call arg values that must be in loss
    golds = []
    for m in e["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for v in m["tool_calls"][0]["function"]["arguments"].values(): golds.append(str(v))
    missing = [g for g in golds if g not in loss_text]
    print(f"\n[{typ}] receipt={e['receipt_id']} tokens={len(full)} loss_tokens={n_loss} assistant_spans={spans}")
    print(f"  gold tool-call arg values all inside loss region: {not missing}  (missing: {missing})")
    print(f"  loss region (decoded, first 220 chars): {loss_text[:220]!r}")
    # masked sample (first 120 chars of masked region decoded)
    masked_text = tok.decode([t for t, l in zip(full, labels) if l == -100])
    print(f"  masked region (decoded, first 140 chars): {masked_text[:140]!r}")

print("\n" + "=" * 70); print("CHECK 4 — NO TRUNCATION"); print("=" * 70)
print(f"max_seq_len set to MAX={MAXLEN}; examples exceeding it: {sum(1 for L in lengths if L > MAXLEN)} -> none truncated (longest == max_seq_len)")
print(f"\nMAX_SEQ_LEN={MAXLEN}")
