#!/usr/bin/env python3
"""Build the runnable authorised-email eval manifest (prereg section 6).

Merges runs/phase4_authorised_email_eval.jsonl (the 20 user-email-request prompt
specs: user_task, expected_to, expected_subject) with the committed CLEAN rows of
runs/phase4_manifest.jsonl (image/transcript paths + sha), producing manifest rows
the harness can run directly: each carries the per-row `user_task` (consumed by the
patched build_model_input) plus expected_to/expected_subject for phase4/score_u2.py.
Text channel, clean receipts. Uses the committed transcripts; never re-renders."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main():
    specs = jl(REPO / "runs" / "phase4_authorised_email_eval.jsonl")
    manifest = jl(REPO / "runs" / "phase4_manifest.jsonl")
    clean_by_receipt = {r["receipt_id"]: r for r in manifest if r["variant"] == "clean"}
    rows = []
    for s in specs:
        base = clean_by_receipt[s["receipt_id"]]
        row = dict(base)                       # committed clean manifest row (paths + sha)
        row["user_task"] = s["user_task"]      # per-row override consumed by build_model_input
        row["expected_to"] = s["expected_to"]
        row["expected_subject"] = s["expected_subject"]
        row["condition"] = "authorised_email"
        rows.append(row)
    out = REPO / "runs" / "phase4_authorised_email_manifest.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out.relative_to(REPO)}")
    print("sample row:", json.dumps(rows[0], ensure_ascii=False)[:300])

if __name__ == "__main__":
    main()
