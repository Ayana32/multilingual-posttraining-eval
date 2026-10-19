#!/usr/bin/env python3
"""Read-only analysis of Phase 2B records.jsonl files. No mutation."""
import json
from pathlib import Path

FILES = {
    ("dpo", "en"): "~/results/phase2b_dpo_en_polyguard_17c33a34/records.jsonl",
    ("dpo", "ko"): "~/results/phase2b_dpo_ko_polyguard_7b95ba0c/records.jsonl",
    ("rlvr", "en"): "~/results/phase2b_rlvr_en_polyguard_151e7314/records.jsonl",
    ("rlvr", "ko"): "~/results/phase2b_rlvr_ko_polyguard_324c0cc3/records.jsonl",
}

rows = []
parse_fail_details = []

for (stage, lang), path_str in FILES.items():
    path = Path(path_str).expanduser()
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    n_match = sum(1 for r in records if r.get("behavior_matches_expected") is True)
    n_match_denom = sum(1 for r in records if r.get("behavior_matches_expected") is not None)

    harmful = [r for r in records if r.get("scorer_prompt_harmful") is True]
    n_refuse_harmful = sum(1 for r in harmful if r.get("scorer_response_refusal") is True)

    unharmful = [r for r in records if r.get("scorer_prompt_harmful") is False]
    n_comply_unharmful = sum(1 for r in unharmful if r.get("scorer_response_refusal") is False)

    n_lang_match = sum(1 for r in records if r.get("language_match") is True)
    n_lang_denom = sum(1 for r in records if r.get("language_match") is not None)

    n_truncated = sum(1 for r in records if r.get("finish_reason") == "length")

    parse_fails = [r for r in records if r.get("scorer_parse_ok") is False]
    for r in parse_fails:
        parse_fail_details.append((stage, lang, r.get("item_id"), r.get("parallel_item_id")))

    rows.append({
        "stage": stage,
        "lang": lang,
        "n": n,
        "accuracy": n_match / n_match_denom if n_match_denom else None,
        "accuracy_denom": n_match_denom,
        "refusal_rate_harmful": n_refuse_harmful / len(harmful) if harmful else None,
        "n_harmful": len(harmful),
        "n_refuse_harmful": n_refuse_harmful,
        "compliance_rate_unharmful": n_comply_unharmful / len(unharmful) if unharmful else None,
        "n_unharmful": len(unharmful),
        "n_comply_unharmful": n_comply_unharmful,
        "language_match_rate": n_lang_match / n_lang_denom if n_lang_denom else None,
        "n_truncated": n_truncated,
        "n_parse_fail": len(parse_fails),
    })

print(f"{'stage':<6}{'lang':<5}{'n':>5}  {'accuracy':>10}  {'refusal|harm':>14}  {'comply|unharm':>15}  {'lang_match':>11}  {'truncated':>10}  {'parse_fail':>11}")
for r in rows:
    acc = f"{r['accuracy']:.1%}" if r['accuracy'] is not None else "n/a"
    ref = f"{r['refusal_rate_harmful']:.1%} ({r['n_refuse_harmful']}/{r['n_harmful']})" if r['refusal_rate_harmful'] is not None else "n/a"
    comp = f"{r['compliance_rate_unharmful']:.1%} ({r['n_comply_unharmful']}/{r['n_unharmful']})" if r['compliance_rate_unharmful'] is not None else "n/a"
    lm = f"{r['language_match_rate']:.1%}" if r['language_match_rate'] is not None else "n/a"
    print(f"{r['stage']:<6}{r['lang']:<5}{r['n']:>5}  {acc:>10}  {ref:>14}  {comp:>15}  {lm:>11}  {r['n_truncated']:>10}  {r['n_parse_fail']:>11}")

print()
if parse_fail_details:
    print(f"scorer_parse_ok=false records ({len(parse_fail_details)}):")
    for stage, lang, item_id, pid in parse_fail_details:
        print(f"  {stage}/{lang}  item_id={item_id}  parallel_item_id={pid}")
else:
    print("No scorer_parse_ok=false records found.")
