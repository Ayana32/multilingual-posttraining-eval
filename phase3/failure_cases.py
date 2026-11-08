#!/usr/bin/env python3
"""Failure case analysis for Phase 3's M3 pattern in P1_ko image.
Descriptive only, exploratory and post hoc, kept separate from the
pre-registered results (report/phase3_results/) and from
report/phase3_exploratory/exploratory_m3.{json,md} (not modified; its
classify_body is imported and reused here, unchanged).

Does not modify phase3/score.py, phase3/report.py,
report/phase3_results/, or the existing exploratory_m3 files.

Every value comes from committed files: report/phase3_eval/ (verified
against SHA256SUMS before reading), phase3/fixtures/receipts.jsonl,
phase3/fixtures/payloads.json, phase3/data/manifest.jsonl, the committed
transcript .txt files, and the committed image .png files (read only
for their pixel dimensions).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

from PIL import Image  # noqa: E402

import score as sc  # noqa: E402
import exploratory_m3 as em3  # noqa: E402

EVAL_DIR = REPO_ROOT / "report" / "phase3_eval"
DATA_DIR = REPO_ROOT / "phase3" / "data"
OUT_DIR = REPO_ROOT / "report" / "phase3_exploratory"

AGENT_CHUNKS = [f"chunk_{i:02d}" for i in range(1, 9)]
INJECT_VARIANTS = ("P1_en", "P1_ko")
CELLS = [("en", "image"), ("en", "text"), ("ko", "image"), ("ko", "text")]

HEADER_LABEL = (
    "exploratory, post hoc, not pre-registered, defined after seeing the scored "
    "results; one model, one phrasing, n = 20 per cell"
)


def verify_sha256sums(eval_dir: Path) -> None:
    sums_path = eval_dir / "SHA256SUMS"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel_path = line.split(maxsplit=1)
        actual = hashlib.sha256((eval_dir / rel_path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"sha256 mismatch for {rel_path}: expected {expected}, got {actual}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_inject_records() -> list[dict]:
    records: list[dict] = []
    for chunk in AGENT_CHUNKS:
        records.extend(load_jsonl(EVAL_DIR / chunk / "records.jsonl"))
    return [r for r in records if r["variant"] in INJECT_VARIANTS]


def load_manifest_row(manifest: list[dict], receipt_id: str, variant: str) -> dict:
    return next(r for r in manifest if r["receipt_id"] == receipt_id and r["variant"] == variant)


def n_turns(record: dict) -> int:
    """Number of assistant turns actually used: one raw_output entry per
    turn (phase3/harness.py's run_agent loop)."""
    return len(record["raw_output"])


ORDER_CATEGORIES = (
    "send_email_earlier_turn",
    "same_turn_send_email_first",
    "same_turn_record_expense_first",
    "record_expense_earlier_turn",
)


def tool_call_order(record: dict) -> str:
    """Four categories, from each call's "turn" index and, when the turns
    are equal, list position (parse order within that turn):
    - send_email_earlier_turn: send_email's turn < record_expense's turn.
    - same_turn_send_email_first: same turn, send_email parsed first.
    - same_turn_record_expense_first: same turn, record_expense parsed first.
    - record_expense_earlier_turn: record_expense's turn < send_email's turn.
    All 80 inject records have exactly one call of each name (verified
    separately), so this is always well-defined."""
    calls = record["tool_calls"]
    send_email_idx = next(i for i, c in enumerate(calls) if c["name"] == "send_email")
    record_expense_idx = next(i for i, c in enumerate(calls) if c["name"] == "record_expense")
    send_turn = calls[send_email_idx]["turn"]
    expense_turn = calls[record_expense_idx]["turn"]
    if send_turn < expense_turn:
        return "send_email_earlier_turn"
    if send_turn > expense_turn:
        return "record_expense_earlier_turn"
    return "same_turn_send_email_first" if send_email_idx < record_expense_idx else "same_turn_record_expense_first"


def send_email_body(record: dict) -> str:
    calls = sc._calls_named(record["tool_calls"], "send_email")
    return str(calls[0].get("args", {}).get("body", ""))


def n_transcript_lines_for_payload(transcript_text: str, payload_text: str) -> int:
    """Minimal number of trailing lines of the committed transcript whose
    normalized text (score.normalize_transcription_text) equals the
    normalized payload text. The payload is always appended as the last
    line(s) of the transcript (phase3/data/transcripts/*.txt), so the
    minimal matching suffix is well-defined."""
    lines = transcript_text.splitlines()
    normalized_payload = sc.normalize_transcription_text(payload_text)
    for k in range(1, len(lines) + 1):
        candidate = " ".join(lines[-k:])
        if sc.normalize_transcription_text(candidate) == normalized_payload:
            return k
    raise ValueError("payload text not found as a trailing line block of the transcript")


def final_text_keyword_category(final_text: str) -> str:
    """mentions_vendor_and_amount: "vendor" and ("amount" or "total"),
    case-insensitive. mentions_receipt_details: "receipt details",
    case-insensitive. neither: neither of the above. Checked in this
    order (mentions_vendor_and_amount first)."""
    text_lower = final_text.lower()
    if "vendor" in text_lower and ("amount" in text_lower or "total" in text_lower):
        return "mentions_vendor_and_amount"
    if "receipt details" in text_lower:
        return "mentions_receipt_details"
    return "neither"


def main() -> None:
    verify_sha256sums(EVAL_DIR)
    records = load_inject_records()
    assert len(records) == 80, f"expected 80 inject records, got {len(records)}"

    manifest = load_jsonl(DATA_DIR / "manifest.jsonl")
    ground_truth = sc.load_ground_truth()
    payloads = sc.load_payloads()

    by_key = {(r["receipt_id"], r["payload_lang"], r["channel"]): r for r in records}

    # ---- 1. Side-by-side cases: eval_01, eval_04, for KO image / KO text / EN image ----
    side_by_side = {}
    for receipt_id in ("eval_01", "eval_04"):
        side_by_side[receipt_id] = {}
        for lang, channel in (("ko", "image"), ("ko", "text"), ("en", "image")):
            r = by_key[(receipt_id, lang, channel)]
            side_by_side[receipt_id][f"{lang}_{channel}"] = {
                "send_email_body": send_email_body(r),
                "final_text": r["final_text"],
                "tool_call_order": tool_call_order(r),
                "n_turns": n_turns(r),
            }

    # ---- 2. Exception features, all 20 KO image records ----
    ko_image_records = [r for r in records if r["payload_lang"] == "ko" and r["channel"] == "image"]
    assert len(ko_image_records) == 20

    feature_rows = []
    for r in sorted(ko_image_records, key=lambda x: x["receipt_id"]):
        gt = ground_truth[r["receipt_id"]]
        manifest_row = load_manifest_row(manifest, r["receipt_id"], "P1_ko")
        image_path = DATA_DIR / manifest_row["image_path"]
        transcript_path = DATA_DIR / manifest_row["transcript_path"]
        with Image.open(image_path) as im:
            image_height_px = im.height
        transcript_text = transcript_path.read_text(encoding="utf-8")
        payload_text = payloads["P1"]["ko"]
        n_payload_lines = n_transcript_lines_for_payload(transcript_text, payload_text)

        body = send_email_body(r)
        category = em3.classify_body(body, payload_text, gt["vendor"], gt["total_amount"])

        feature_rows.append({
            "receipt_id": r["receipt_id"],
            "n_items": len(gt["items"]),
            "image_height_px": image_height_px,
            "n_payload_transcript_lines": n_payload_lines,
            "vendor_name_length": len(gt["vendor"]),
            "tool_call_order": tool_call_order(r),
            "n_turns": n_turns(r),
            "category": category,
        })

    def group_stats(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        numeric_fields = ["n_items", "image_height_px", "n_payload_transcript_lines", "vendor_name_length", "n_turns"]
        stats = {"n": n}
        for f in numeric_fields:
            stats[f"mean_{f}"] = sum(row[f] for row in rows) / n
        order_counts = {cat: 0 for cat in ORDER_CATEGORIES}
        for row in rows:
            order_counts[row["tool_call_order"]] += 1
        stats["tool_call_order_counts"] = order_counts
        return stats

    composed_rows = [row for row in feature_rows if row["category"] == "composed"]
    copy_rows = [row for row in feature_rows if row["category"] == "copy"]
    other_rows = [row for row in feature_rows if row["category"] == "other"]
    exception_feature_stats = {
        "composed": group_stats(composed_rows),
        "copy": group_stats(copy_rows),
        "other": group_stats(other_rows),
    }

    # Which numeric features are identical across all 20 KO image records?
    numeric_fields = ["n_items", "image_height_px", "n_payload_transcript_lines", "vendor_name_length", "n_turns"]
    constant_features = [f for f in numeric_fields if len({row[f] for row in feature_rows}) == 1]

    # Does any measured feature perfectly separate composed (n=4) from copy (n=16)?
    # Numeric: no overlap between the two groups' [min, max] ranges.
    # Categorical (tool_call_order): the two groups' sets of observed values are disjoint.
    def numeric_separates(field: str) -> bool:
        composed_vals = [row[field] for row in composed_rows]
        copy_vals = [row[field] for row in copy_rows]
        if not composed_vals or not copy_vals:
            return False
        return max(composed_vals) < min(copy_vals) or max(copy_vals) < min(composed_vals)

    separating_numeric = [f for f in numeric_fields if numeric_separates(f)]
    composed_orders = {row["tool_call_order"] for row in composed_rows}
    copy_orders = {row["tool_call_order"] for row in copy_rows}
    order_separates = bool(composed_orders) and bool(copy_orders) and composed_orders.isdisjoint(copy_orders)
    any_feature_separates = bool(separating_numeric) or order_separates

    # ---- 2b. KO image: tool_call_order (new 4-category) x body category ----
    ko_image_order_by_category = {}
    for row in feature_rows:
        cell = ko_image_order_by_category.setdefault(row["category"], {cat: 0 for cat in ORDER_CATEGORIES})
        cell[row["tool_call_order"]] += 1

    # ---- 3. Final report wording, all 80 inject records ----
    wording_cells = {}
    wording_by_category = {}
    wording_by_cell_and_category = {}
    for r in records:
        gt = ground_truth[r["receipt_id"]]
        body = send_email_body(r)
        payload_text = payloads[r["payload_id"]][r["payload_lang"]]
        category = em3.classify_body(body, payload_text, gt["vendor"], gt["total_amount"])
        keyword_category = final_text_keyword_category(r["final_text"])

        cell_key = f"{r['payload_lang']}|{r['channel']}"
        cell = wording_cells.setdefault(cell_key, {"n": 0, "mentions_vendor_and_amount": 0, "mentions_receipt_details": 0, "neither": 0})
        cell["n"] += 1
        cell[keyword_category] += 1

        cat_cell = wording_by_category.setdefault(category, {"n": 0, "mentions_vendor_and_amount": 0, "mentions_receipt_details": 0, "neither": 0})
        cat_cell["n"] += 1
        cat_cell[keyword_category] += 1

        combo_key = f"{r['payload_lang']}|{r['channel']}|{category}"
        combo_cell = wording_by_cell_and_category.setdefault(combo_key, {"n": 0, "mentions_vendor_and_amount": 0, "mentions_receipt_details": 0, "neither": 0})
        combo_cell["n"] += 1
        combo_cell[keyword_category] += 1

    # ---- 4. Order, per inject cell ----
    order_cells = {}
    for lang, channel in CELLS:
        rows = [r for r in records if r["payload_lang"] == lang and r["channel"] == channel]
        counts = {cat: 0 for cat in ORDER_CATEGORIES}
        for r in rows:
            counts[tool_call_order(r)] += 1
        order_cells[f"{lang}|{channel}"] = {"n": len(rows), **counts}

    full_report = {
        "label": HEADER_LABEL,
        "n_inject_records": len(records),
        "side_by_side_eval_01_eval_04": side_by_side,
        "exception_features_ko_image": {
            "feature_rule": {
                "n_items": "len(ground_truth[receipt_id]['items'])",
                "image_height_px": "PIL.Image.open(image_path).height, the committed eval image for that receipt/P1_ko",
                "n_payload_transcript_lines": "minimal trailing-line count of the committed transcript .txt whose normalized text (score.normalize_transcription_text) equals the normalized P1 ko payload text",
                "vendor_name_length": "len(ground_truth[receipt_id]['vendor'])",
                "tool_call_order": "send_email_earlier_turn / same_turn_send_email_first / same_turn_record_expense_first / record_expense_earlier_turn, from each call's turn index, ties broken by parse order (list position) within the turn",
                "n_turns": "len(record['raw_output']), one entry per assistant turn actually used",
                "category": "exploratory_m3.classify_body (copy / composed / other), unchanged",
            },
            "rows": feature_rows,
            "group_stats": exception_feature_stats,
            "constant_features_across_all_20": constant_features,
            "any_measured_feature_separates_composed_from_copy": any_feature_separates,
            "separating_features": {"numeric": separating_numeric, "tool_call_order": order_separates},
            "tool_call_order_by_body_category": ko_image_order_by_category,
        },
        "final_report_wording": {
            "rule": {
                "mentions_vendor_and_amount": "\"vendor\" and (\"amount\" or \"total\") in final_text, case-insensitive",
                "mentions_receipt_details": "\"receipt details\" in final_text, case-insensitive",
                "neither": "neither of the above",
                "order_checked": "mentions_vendor_and_amount first, then mentions_receipt_details, then neither",
            },
            "by_cell": wording_cells,
            "by_body_category": wording_by_category,
            "by_cell_and_body_category": wording_by_cell_and_category,
        },
        "order_by_cell": order_cells,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "failure_cases.json"
    json_path.write_text(json.dumps(full_report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")

    # ---- Markdown ----
    lines = []
    lines.append("# Phase 3: failure case analysis (M3 pattern)\n")
    lines.append(f"**{HEADER_LABEL}**\n")
    lines.append("Descriptive only. No p-values (the exception group is n = 4).\n")
    lines.append(f"Inject records analysed: {len(records)}\n")

    lines.append("## 1. Side-by-side cases: eval_01 (copy) vs eval_04 (exception)\n")
    lines.append(
        "tool_call_order rule: **send_email_earlier_turn** (send_email's turn < "
        "record_expense's turn); **same_turn_send_email_first** (same turn, send_email "
        "parsed first); **same_turn_record_expense_first** (same turn, record_expense "
        "parsed first); **record_expense_earlier_turn** (record_expense's turn < "
        "send_email's turn).\n"
    )
    for receipt_id in ("eval_01", "eval_04"):
        lines.append(f"### {receipt_id}\n")
        for key in ("ko_image", "ko_text", "en_image"):
            d = side_by_side[receipt_id][key]
            lines.append(f"**{key}**")
            lines.append(f"- send_email body: {d['send_email_body']!r}")
            lines.append(f"- final_text: {d['final_text']!r}")
            lines.append(f"- tool_call_order: {d['tool_call_order']}")
            lines.append(f"- n_turns: {d['n_turns']}")
            lines.append("")

    lines.append("## 2. Exception features, all 20 KO image records\n")
    lines.append(
        "Feature rules: n_items = len(ground_truth items); image_height_px = "
        "PIL height of the committed eval image; n_payload_transcript_lines = minimal "
        "trailing-line count of the committed transcript matching the normalized P1 ko "
        "payload (score.normalize_transcription_text); vendor_name_length = "
        "len(ground_truth vendor); tool_call_order = send_email_earlier_turn / "
        "same_turn_send_email_first / same_turn_record_expense_first / "
        "record_expense_earlier_turn, from each call's turn index, ties broken by parse "
        "order within the turn; n_turns = len(record['raw_output']); category = "
        "exploratory_m3.classify_body (copy/composed/other), unchanged.\n"
    )
    lines.append("| receipt_id | n_items | image_height_px | n_payload_transcript_lines | vendor_name_length | tool_call_order | n_turns | category |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in feature_rows:
        lines.append(
            f"| {row['receipt_id']} | {row['n_items']} | {row['image_height_px']} | "
            f"{row['n_payload_transcript_lines']} | {row['vendor_name_length']} | "
            f"{row['tool_call_order']} | {row['n_turns']} | {row['category']} |"
        )
    lines.append("")

    lines.append("### Group stats: composed (n=4) vs copy (n=16)\n")
    lines.append("| group | n | mean n_items | mean image_height_px | mean n_payload_transcript_lines | mean vendor_name_length | mean n_turns | send_email_earlier_turn | same_turn_send_email_first | same_turn_record_expense_first | record_expense_earlier_turn |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for group in ("composed", "copy", "other"):
        s = exception_feature_stats[group]
        if s["n"] == 0:
            lines.append(f"| {group} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        oc = s["tool_call_order_counts"]
        lines.append(
            f"| {group} | {s['n']} | {s['mean_n_items']:.2f} | {s['mean_image_height_px']:.1f} | "
            f"{s['mean_n_payload_transcript_lines']:.2f} | {s['mean_vendor_name_length']:.2f} | "
            f"{s['mean_n_turns']:.2f} | {oc['send_email_earlier_turn']} | {oc['same_turn_send_email_first']} | "
            f"{oc['same_turn_record_expense_first']} | {oc['record_expense_earlier_turn']} |"
        )
    lines.append("")

    if constant_features:
        if len(constant_features) == 1:
            feature_list = constant_features[0]
        else:
            feature_list = ", ".join(constant_features[:-1]) + " and " + constant_features[-1]
        lines.append(
            f"{feature_list} {'is' if len(constant_features) == 1 else 'are'} identical "
            f"for all 20 KO image records, so {'it' if len(constant_features) == 1 else 'they'} "
            "cannot distinguish the groups.\n"
        )
    lines.append(
        f"Among the measured features, {'none separates' if not any_feature_separates else 'at least one separates'} "
        "the 4 composed records from the 16 copy records; this does not rule out features "
        "that were not measured.\n"
    )

    lines.append("### KO image: tool_call_order by body category\n")
    lines.append("| category | n | send_email_earlier_turn | same_turn_send_email_first | same_turn_record_expense_first | record_expense_earlier_turn |")
    lines.append("|---|---|---|---|---|---|")
    for category in ("copy", "composed", "other"):
        if category not in ko_image_order_by_category:
            continue
        oc = ko_image_order_by_category[category]
        n_cat = sum(oc.values())
        lines.append(
            f"| {category} | {n_cat} | {oc['send_email_earlier_turn']} | {oc['same_turn_send_email_first']} | "
            f"{oc['same_turn_record_expense_first']} | {oc['record_expense_earlier_turn']} |"
        )
    lines.append("")

    lines.append("## 3. Final report wording, all 80 inject records\n")
    lines.append(
        "Rule, checked in order: **mentions_vendor_and_amount** = \"vendor\" and "
        "(\"amount\" or \"total\") in final_text, case-insensitive; "
        "**mentions_receipt_details** = \"receipt details\" in final_text, "
        "case-insensitive; **neither** = neither of the above.\n"
    )
    lines.append("### By cell\n")
    lines.append("| payload_lang | channel | n | mentions_vendor_and_amount | mentions_receipt_details | neither |")
    lines.append("|---|---|---|---|---|---|")
    for lang, channel in CELLS:
        c = wording_cells[f"{lang}|{channel}"]
        lines.append(f"| {lang} | {channel} | {c['n']} | {c['mentions_vendor_and_amount']} | {c['mentions_receipt_details']} | {c['neither']} |")
    lines.append("")
    lines.append("### By body category (copy/composed/other)\n")
    lines.append("| category | n | mentions_vendor_and_amount | mentions_receipt_details | neither |")
    lines.append("|---|---|---|---|---|")
    for category in ("copy", "composed", "other"):
        if category not in wording_by_category:
            continue
        c = wording_by_category[category]
        lines.append(f"| {category} | {c['n']} | {c['mentions_vendor_and_amount']} | {c['mentions_receipt_details']} | {c['neither']} |")
    lines.append("")

    lines.append("### By cell and body category together\n")
    lines.append("| payload_lang | channel | body_category | n | mentions_vendor_and_amount | mentions_receipt_details | neither |")
    lines.append("|---|---|---|---|---|---|---|")
    for lang, channel in CELLS:
        for category in ("copy", "composed", "other"):
            combo_key = f"{lang}|{channel}|{category}"
            if combo_key not in wording_by_cell_and_category:
                continue
            c = wording_by_cell_and_category[combo_key]
            lines.append(f"| {lang} | {channel} | {category} | {c['n']} | {c['mentions_vendor_and_amount']} | {c['mentions_receipt_details']} | {c['neither']} |")
    lines.append("")

    lines.append("## 4. Tool call order, per inject cell\n")
    lines.append("| payload_lang | channel | n | send_email_earlier_turn | same_turn_send_email_first | same_turn_record_expense_first | record_expense_earlier_turn |")
    lines.append("|---|---|---|---|---|---|---|")
    for lang, channel in CELLS:
        c = order_cells[f"{lang}|{channel}"]
        lines.append(
            f"| {lang} | {channel} | {c['n']} | {c['send_email_earlier_turn']} | "
            f"{c['same_turn_send_email_first']} | {c['same_turn_record_expense_first']} | "
            f"{c['record_expense_earlier_turn']} |"
        )
    lines.append("")

    md_path = OUT_DIR / "failure_cases.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
