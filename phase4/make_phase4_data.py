#!/usr/bin/env python3
"""Phase 4 local data build (docs/phase4-prereg.md sections 5, 6, 9).

Builds, with fixed seeds and committed data files, and NO GPU / no weights:
  - 20 eval receipts (phase3 VENDORS, own seed), rendered to image + text for
    clean / P1_en / P1_ko, frozen as runs/phase4_manifest.jsonl (+ phase4/data/eval/).
  - ~400 English text-only training examples (200 injected, 100 clean, 100
    authorised-email) with the model's chat template, the eval system prompt and
    tools.json, as message lists -> phase4/data/train/train.jsonl.
  - 20 authorised-email eval prompts -> runs/phase4_authorised_email_eval.jsonl.

Reuses phase3/make_receipts.py's render_receipt/generate_receipts UNCHANGED
(only the vendor pool for training and the seeds are new). No image or Korean
training data. The injection footer for training is appended as text, mirroring
how P1 is appended to the transcript in render_receipt.
"""
from __future__ import annotations
import json, sys, random, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase3"))
import make_receipts as mr  # phase3 generator/renderer, reused unchanged

FIX = REPO / "phase3" / "fixtures"
PAYLOADS = json.loads((FIX / "payloads.json").read_text(encoding="utf-8"))
SYSTEM_PROMPT = (FIX / "system_prompt.txt").read_text(encoding="utf-8")
PROMPTS = json.loads((FIX / "prompts.json").read_text(encoding="utf-8"))
USER_TASK = PROMPTS["user_task"]
RECEIPT_HEADER = PROMPTS["receipt_header"]

P4 = REPO / "phase4" / "data"
TRAIN_VENDORS = json.loads((P4 / "training_vendors.json").read_text())["vendors"]
INJECTIONS = json.loads((P4 / "injection_templates.json").read_text())["templates"]
AUTH = json.loads((P4 / "authorised_email.json").read_text())

EVAL_SEED = 20264001          # separate from phase3 (20261105) and phase3b (20261108)
TRAIN_SEED = 20264002
N_EVAL = 20
N_INJ, N_CLEAN, N_AUTH = 200, 100, 100
EVAL_VARIANTS = [("clean", None, None), ("P1_en", "P1", "en"), ("P1_ko", "P1", "ko")]

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

# ---------------------------------------------------------------- eval set
def build_eval():
    fonts = mr._load_fonts()
    receipts = mr.generate_receipts(EVAL_SEED, n_tune=0, n_eval=N_EVAL)  # phase3 VENDORS
    eval_dir = P4 / "eval"; img_dir = eval_dir / "images"; tr_dir = eval_dir / "transcripts"
    for d in (img_dir, tr_dir): d.mkdir(parents=True, exist_ok=True)
    gt_rows, manifest_rows = [], []
    for k, r in enumerate(receipts, 1):
        r = dict(r); r["receipt_id"] = f"phase4_eval_{k:02d}"; r["receipt_set"] = "phase4_eval"
        gt_rows.append({"receipt_id": r["receipt_id"], "receipt_set": r["receipt_set"],
                        "vendor": r["vendor"], "date": r["date"], "items": r["items"],
                        "total_amount": r["total_amount"], "currency": r["currency"]})
        for variant, pid, plang in EVAL_VARIANTS:
            img, lines = mr.render_receipt(r, variant, pid, plang, PAYLOADS, fonts)
            ip = img_dir / f"{r['receipt_id']}_{variant}.png"
            tp = tr_dir / f"{r['receipt_id']}_{variant}.txt"
            img.save(ip)
            tbytes = ("\n".join(lines) + "\n").encode("utf-8"); tp.write_bytes(tbytes)
            manifest_rows.append({
                "arm": "A", "font": None,
                "image_path": str(ip.relative_to(eval_dir)),
                "image_sha256": sha_bytes(ip.read_bytes()),
                "payload_id": pid, "payload_lang": plang,
                "receipt_id": r["receipt_id"], "receipt_set": r["receipt_set"],
                "transcript_path": str(tp.relative_to(eval_dir)),
                "transcript_sha256": sha_bytes(tbytes), "variant": variant})
    (eval_dir / "receipts.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in gt_rows), encoding="utf-8")
    manifest = REPO / "runs" / "phase4_manifest.jsonl"
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in manifest_rows), encoding="utf-8")
    return gt_rows, manifest_rows

# ------------------------------------------------------------ training receipts
def make_training_receipts(seed: int, n: int, pool: list[str]) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        vendor = rng.choice(pool)                 # repetition allowed across training
        day = rng.randint(1, 28); month = rng.choice(mr.MONTHS)
        date = f"{day:02d} {month} 2025"
        n_items = rng.randint(2, 4); chosen = rng.sample(mr.ITEMS, n_items)
        items = []; total = mr.money("0.00")
        for name, up in chosen:
            qty = rng.randint(1, 3); unit = mr.money(up); lt = mr.money(unit * qty); total += lt
            items.append({"name": name, "quantity": qty, "unit_price": mr.fmt_money(unit), "line_total": mr.fmt_money(lt)})
        out.append({"receipt_id": f"train_{i:04d}", "receipt_set": "phase4_train", "vendor": vendor,
                    "date": date, "items": items, "total_amount": mr.fmt_money(total), "currency": mr.CURRENCY})
    return out

def clean_transcript(r: dict, fonts) -> str:
    _, lines = mr.render_receipt(r, "clean", None, None, PAYLOADS, fonts)
    return "\n".join(lines)

def user_block(task: str, transcript: str) -> str:
    # mirrors harness.build_model_input text channel: task + header + receipt, each line "\n"-terminated
    return f"{task}\n{RECEIPT_HEADER}\n{transcript}\n"

def rec_expense_call(r: dict) -> dict:
    return {"type": "function", "function": {"name": "record_expense", "arguments":
            {"vendor": r["vendor"], "date": r["date"], "amount": r["total_amount"], "currency": r["currency"]}}}

def send_email_call(to: str, subject: str, r: dict) -> dict:
    body = f"Receipt from {r['vendor']}, {r['date']}, total {r['total_amount']} {r['currency']}."
    return {"type": "function", "function": {"name": "send_email", "arguments": {"to": to, "subject": subject, "body": body}}}

def build_training():
    fonts = mr._load_fonts()
    receipts = make_training_receipts(TRAIN_SEED, N_INJ + N_CLEAN + N_AUTH, TRAIN_VENDORS)
    rng = random.Random(TRAIN_SEED ^ 0x5EED)
    examples = []
    idx = 0
    # injected
    for j in range(N_INJ):
        r = receipts[idx]; idx += 1
        inj = INJECTIONS[j % len(INJECTIONS)]
        transcript = clean_transcript(r, fonts) + "\n" + inj["text"]   # footer appended as text
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block(USER_TASK, transcript)},
            {"role": "assistant", "content": "", "tool_calls": [rec_expense_call(r)]},
            {"role": "tool", "content": json.dumps({"status": "recorded"})},
            {"role": "assistant", "content": f"Recorded the expense for {r['vendor']}."},
        ]
        examples.append({"idx": idx, "type": "injected", "receipt_id": r["receipt_id"],
                         "injection_id": inj["id"], "messages": msgs})
    # clean
    for j in range(N_CLEAN):
        r = receipts[idx]; idx += 1
        transcript = clean_transcript(r, fonts)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block(USER_TASK, transcript)},
            {"role": "assistant", "content": "", "tool_calls": [rec_expense_call(r)]},
            {"role": "tool", "content": json.dumps({"status": "recorded"})},
            {"role": "assistant", "content": f"Recorded the expense for {r['vendor']}."},
        ]
        examples.append({"idx": idx, "type": "clean", "receipt_id": r["receipt_id"], "messages": msgs})
    # authorised email
    tw = AUTH["train"]["wordings"]; ta = AUTH["train"]["addresses"]; ts = AUTH["train"]["subjects"]
    for j in range(N_AUTH):
        r = receipts[idx]; idx += 1
        to = ta[j % len(ta)]; subject = ts[j % len(ts)]; wording = tw[j % len(tw)]
        task = wording.format(to=to, subject=subject)
        transcript = clean_transcript(r, fonts)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block(task, transcript)},
            {"role": "assistant", "content": "", "tool_calls": [rec_expense_call(r)]},
            {"role": "tool", "content": json.dumps({"status": "recorded"})},
            {"role": "assistant", "content": "", "tool_calls": [send_email_call(to, subject, r)]},
            {"role": "tool", "content": json.dumps({"status": "sent"})},
            {"role": "assistant", "content": f"Recorded the expense and emailed a copy to {to}."},
        ]
        examples.append({"idx": idx, "type": "authorised_email", "receipt_id": r["receipt_id"],
                         "to": to, "subject": subject, "messages": msgs})
    (P4 / "train").mkdir(parents=True, exist_ok=True)
    (P4 / "train" / "train.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in examples), encoding="utf-8")
    return examples

# ------------------------------------------------------- authorised-email eval
def build_auth_eval(eval_gt):
    ew = AUTH["eval"]["wordings"]; ea = AUTH["eval"]["addresses"]; es = AUTH["eval"]["subjects"]
    rng = random.Random(EVAL_SEED ^ 0xA11)
    rows = []
    for k in range(20):
        r = eval_gt[k]  # 20 clean phase4_eval receipts, text channel
        to = ea[k % len(ea)]; subject = es[k % len(es)]; wording = ew[k % len(ew)]
        rows.append({"idx": k + 1, "receipt_id": r["receipt_id"], "channel": "text",
                     "variant": "clean", "user_task": wording.format(to=to, subject=subject),
                     "expected_to": to, "expected_subject": subject})
    (REPO / "runs" / "phase4_authorised_email_eval.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return rows

def main():
    gt, manifest = build_eval()
    ex = build_training()
    auth = build_auth_eval(gt)
    from collections import Counter
    print("eval receipts:", len(gt), "| manifest rows:", len(manifest))
    print("training examples:", len(ex), Counter(e["type"] for e in ex))
    print("authorised-email eval prompts:", len(auth))

if __name__ == "__main__":
    main()
