#!/usr/bin/env python3
"""Phase 3 R_eval scoring driver. Loads R_eval records from
report/phase3_eval/ (the committed copy; verify against
report/phase3_eval/SHA256SUMS before running this), and calls
phase3/score.py and phase3/report.py's existing functions -- no new
measures or aggregation formulas are defined here, only loading,
dispatch to those functions, and tabulation of their output. Writes
report/phase3_results/phase3_results.json and .md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import score as sc  # noqa: E402
import report as rpt  # noqa: E402

EVAL_DIR = REPO_ROOT / "report" / "phase3_eval"
OUT_DIR = REPO_ROOT / "report" / "phase3_results"

AGENT_CHUNKS = [f"chunk_{i:02d}" for i in range(1, 9)]
TRANSCRIPTION_CHUNKS = [f"chunk_transcription_{i:02d}" for i in range(1, 3)]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    agent_records: list[dict] = []
    for chunk in AGENT_CHUNKS:
        agent_records.extend(load_jsonl(EVAL_DIR / chunk / "records.jsonl"))

    transcription_records: list[dict] = []
    for chunk in TRANSCRIPTION_CHUNKS:
        transcription_records.extend(load_jsonl(EVAL_DIR / chunk / "transcription_records.jsonl"))

    print(f"Loaded {len(agent_records)} agent records, {len(transcription_records)} transcription records")

    ground_truth = sc.load_ground_truth()
    payloads = sc.load_payloads()

    scores = sc.score_records(agent_records, ground_truth)
    report = rpt.build_report(scores)

    transcription_scores = [sc.score_transcription_record(r, payloads) for r in transcription_records]
    transcription_out = [
        {
            "receipt_id": t.receipt_id,
            "receipt_set": t.receipt_set,
            "variant": t.variant,
            "payload_id": t.payload_id,
            "payload_lang": t.payload_lang,
            "exact_match": t.exact_match,
            "cer": t.cer,
            "legibility_label": t.legibility_label,
        }
        for t in transcription_scores
    ]

    full_report = {
        "n_agent_records": len(agent_records),
        "n_transcription_records": len(transcription_records),
        "primary_measures": report,
        "transcription_T": transcription_out,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "phase3_results.json"
    json_path.write_text(json.dumps(full_report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")

    # ---- Markdown table (tabulation only, no new measures) ----
    lines = []
    lines.append("# Phase 3 R_eval scoring (pre-registered measures only)\n")
    lines.append(f"Agent records: {len(agent_records)}  Transcription records: {len(transcription_records)}\n")

    lines.append("## Primary measures (M1, M2, M3, TASK, CO) per condition\n")
    lines.append("| variant | channel | payload_lang | n | M1 k/n (95% Wilson) | M2 k/n (95% Wilson) | M3 k/n (95% Wilson) | TASK k/n (95% Wilson) | CO k/n (95% Wilson) | worst_case_M1 k/n |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    def fmt_cell(entry, measure):
        k, n = entry[measure]["k"], entry[measure]["n"]
        if n == 0:
            return "n/a"
        lo, hi = entry[measure]["wilson_low"], entry[measure]["wilson_high"]
        return f"{k}/{n} ({lo:.3f}, {hi:.3f})"

    for key in sorted(report["cells"].keys()):
        variant, channel, payload_lang = key.split("|")
        entry = report["cells"][key]
        n = entry["n"]
        row = [variant, channel, payload_lang, str(n)]
        for measure in ("M1", "M2", "M3", "TASK", "CO"):
            row.append(fmt_cell(entry, measure) if n > 0 else "n/a")
        wc = entry["worst_case_M1"]
        row.append(f"{wc['k']}/{wc['n']}" if n > 0 else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## False positives: M1 in clean/neutral variants\n")
    lines.append("| variant | channel | payload_lang | M1 k/n |")
    lines.append("|---|---|---|---|")
    for key in sorted(report["false_positives"].keys()):
        variant, channel, payload_lang = key.split("|")
        fp = report["false_positives"][key]
        lines.append(f"| {variant} | {channel} | {payload_lang} | {fp['M1']}/{fp['n']} |")

    lines.append("\n## Paired EN/KO discordant counts (M2), per channel x payload_id\n")
    lines.append("| channel | payload_id | n_pairs | en1_ko0 | en0_ko1 | both | neither |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(report["paired_en_ko_discordant"].keys()):
        channel, payload_id = key.split("|")
        d = report["paired_en_ko_discordant"][key]
        lines.append(f"| {channel} | {payload_id} | {d['n_pairs']} | {d['en1_ko0']} | {d['en0_ko1']} | {d['both']} | {d['neither']} |")

    lines.append("\n## d_i = (EN_image - KO_image) - (EN_text - KO_text), per payload_id (M2)\n")
    lines.append("| payload_id | n | mean(d_i) | values |")
    lines.append("|---|---|---|---|")
    for payload_id in sorted(report["d_i"].keys()):
        d = report["d_i"][payload_id]
        n = d["n"]
        mean = d["mean"]
        mean_str = "n/a" if mean is None else f"{mean:.3f}"
        values_str = ", ".join(str(v) for v in d["values"]) if d["values"] else "n/a"
        lines.append(f"| {payload_id} | {n} | {mean_str} | {values_str} |")

    lines.append("\n## Excluded records (harness_error, failed twice)\n")
    if report["excluded"]:
        lines.append("| receipt_id | receipt_set | variant | channel |")
        lines.append("|---|---|---|---|")
        for e in report["excluded"]:
            lines.append(f"| {e['receipt_id']} | {e['receipt_set']} | {e['variant']} | {e['channel']} |")
    else:
        lines.append("None.\n")

    lines.append("\n## T (transcription): exact match, CER, per record\n")
    lines.append("| receipt_id | variant | payload_id | payload_lang | exact_match | cer | legibility_label |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in sorted(transcription_out, key=lambda x: (x["payload_id"], x["payload_lang"], x["receipt_id"])):
        lines.append(f"| {t['receipt_id']} | {t['variant']} | {t['payload_id']} | {t['payload_lang']} | {t['exact_match']} | {t['cer']:.4f} | {t['legibility_label']} |")

    md_path = OUT_DIR / "phase3_results.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
