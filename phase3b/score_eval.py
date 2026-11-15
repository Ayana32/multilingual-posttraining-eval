#!/usr/bin/env python3
"""Phase 3b R_eval scoring driver (docs/phase3b-replication.md Sections 5-9).

Loads the chunk outputs produced by phase3b/run.py (records.jsonl /
transcription_records.jsonl under an input dir), splits them by arm (and arm B
by font), and dispatches to phase3b/score.py, phase3/score.py and
phase3/report.py -- no new measures are defined here, only loading, dispatch and
tabulation. Writes report/phase3b_results/phase3b_results.{json,md}.

Ground truth is phase3b/data/receipts.jsonl (the Phase 3b receipts), payloads
are phase3/fixtures/payloads.json (reused).

Sections covered: 4/6 copy + conditional + copy_first_call + worst_case +
category per cell; 5 H1 (one-sided exact McNemar for P1_ko image>text) with the
Newcombe and bootstrap intervals; 6 P2_ko (descriptive), M1-M3/TASK/CO per cell
with Wilson, false positives, paired EN/KO discordant, d_i on M2 and on copy, T;
7 arm B copy per font + the agent-output determinism check vs arm A; 8 arm C
M1/M2/copy/TASK and both d_i measures next to arm A.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import score as sc  # noqa: E402
import report as rpt  # noqa: E402
import exploratory_m3 as em  # noqa: E402  (verify_sha256sums, reused)
import failure_cases as fc  # noqa: E402  (n_transcript_lines_for_payload, reused)

import importlib.util  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p3s = _load("phase3b_score", "phase3b/score.py")

DATA_DIR = REPO_ROOT / "phase3b" / "data"
# Scoring reads ONLY the committed, SHA-verified report/phase3b_eval/ (written by
# phase3b/publish_eval.py), never the raw gitignored pod results.
DEFAULT_INPUT_DIR = REPO_ROOT / "report" / "phase3b_eval"
OUT_DIR = REPO_ROOT / "report" / "phase3b_results"

# Section 9 totals and the pinned serving stack (docs/phase3b-replication.md
# Section 2). The merge check refuses anything that does not match.
EXPECTED_AGENT = 960
EXPECTED_TRANSCRIPTION = 400
PINNED_VERSIONS = {"transformers_version": "4.57.1", "torch_version": "2.8.0+cu128"}


class MergeCheckError(RuntimeError):
    """Raised when the published records do not match the Section 9 run table or
    the frozen environment (wrong count, a duplicate cell, or a chunk whose
    environment.json is dirty or off the pinned versions)."""


def check_record_counts(agent_records: list[dict], transcription_records: list[dict]) -> None:
    if len(agent_records) != EXPECTED_AGENT:
        raise MergeCheckError(f"expected {EXPECTED_AGENT} agent records, got {len(agent_records)}")
    if len(transcription_records) != EXPECTED_TRANSCRIPTION:
        raise MergeCheckError(f"expected {EXPECTED_TRANSCRIPTION} transcription records, got {len(transcription_records)}")
    seen: dict[tuple, int] = {}
    for r in agent_records:
        key = (r.get("arm"), r.get("font"), r.get("channel"), r["receipt_id"], r["variant"])
        seen[key] = seen.get(key, 0) + 1
    dups = sorted(k for k, n in seen.items() if n > 1)
    if dups:
        raise MergeCheckError(f"duplicate (arm, font, channel, receipt_id, variant) cells: {dups[:5]}"
                              f"{' ...' if len(dups) > 5 else ''}")


def check_chunk_environments(eval_dir: Path) -> None:
    env_paths = sorted(eval_dir.rglob("environment.json"))
    if not env_paths:
        raise MergeCheckError(f"no chunk environment.json found under {eval_dir}")
    for path in env_paths:
        env = json.loads(path.read_text(encoding="utf-8"))
        if env.get("git_dirty") is not False:
            raise MergeCheckError(f"{path.parent.name}/environment.json git_dirty is {env.get('git_dirty')!r}, not false")
        for key, want in PINNED_VERSIONS.items():
            if env.get(key) != want:
                raise MergeCheckError(f"{path.parent.name}/environment.json {key} is {env.get(key)!r}, not {want!r}")


def merge_check(agent_records: list[dict], transcription_records: list[dict], eval_dir: Path) -> None:
    """The Section 9 merge check as code: counts, no duplicate cell, and every
    chunk's environment.json clean and on the pinned versions. Refuses
    otherwise, before any scoring."""
    check_record_counts(agent_records, transcription_records)
    check_chunk_environments(eval_dir)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_records(input_dir: Path, verify: bool = True) -> tuple[list[dict], list[dict]]:
    """Verifies input_dir/SHA256SUMS first (unless verify=False), then walks it
    for every records.jsonl (agent) and transcription_records.jsonl
    (transcription), in sorted order."""
    if verify:
        em.verify_sha256sums(input_dir)
    agent, transcription = [], []
    for path in sorted(input_dir.rglob("records.jsonl")):
        agent.extend(load_jsonl(path))
    for path in sorted(input_dir.rglob("transcription_records.jsonl")):
        transcription.extend(load_jsonl(path))
    return agent, transcription


def _variant_payload(variant: str) -> tuple[str | None, str | None]:
    if variant == "clean":
        return None, None
    prefix, _, lang = variant.partition("_")
    return prefix, lang  # e.g. ("P1", "en"), ("neutral", "ko")


def _cell_footer_lines(data_dir: Path, payloads: dict, arm: str, font: str | None, variant: str) -> int | None:
    """Footer line count for a (arm, font, variant) cell, read from a
    representative committed transcript (phase3b_eval_01) with
    phase3/failure_cases.py::n_transcript_lines_for_payload. None for clean (no
    footer) or when the transcript is absent."""
    payload_id, lang = _variant_payload(variant)
    if payload_id is None:
        return None
    rid = "phase3b_eval_01"
    stem = f"{rid}_{variant}_{font}" if arm == "B" else f"{rid}_{variant}"
    tp = data_dir / "transcripts" / f"{stem}.txt"
    if not tp.exists():
        return None
    return fc.n_transcript_lines_for_payload(tp.read_text(encoding="utf-8"), payloads[payload_id][lang])


_ARM_A_VARIANTS = ["clean", "neutral_en", "neutral_ko", "P1_en", "P1_ko", "P2_en", "P2_ko"]
_ARM_B_VARIANTS = ["P1_en", "P1_ko"]
_ARM_C_VARIANTS = ["clean", "P1_en", "P1_ko"]


def _footer_line_counts(data_dir: Path, payloads: dict) -> dict:
    """{'arm|font|variant': footer_line_count} over the full canonical cell set
    (a rendering property, independent of which eval records exist)."""
    out: dict[str, int | None] = {}
    for v in _ARM_A_VARIANTS:
        out[f"A|-|{v}"] = _cell_footer_lines(data_dir, payloads, "A", None, v)
    for font in ("NotoSansKR", "NanumGothic"):
        for v in _ARM_B_VARIANTS:
            out[f"B|{font}|{v}"] = _cell_footer_lines(data_dir, payloads, "B", font, v)
    for v in _ARM_C_VARIANTS:
        out[f"C|-|{v}"] = _cell_footer_lines(data_dir, payloads, "C", None, v)
    return out


def _paired(cp_scores, payload_id, payload_lang) -> dict:
    table = p3s.paired_copy_table(cp_scores, payload_id, payload_lang)
    out = {"table": table, "mcnemar": p3s.h1_test(table),
           "bootstrap": p3s.bootstrap_paired_difference(cp_scores, payload_id, payload_lang)}
    out["newcombe"] = p3s.newcombe_interval(
        table["n11"], table["n10"], table["n01"], table["n00"]) if table["n"] > 0 else None
    return out


def _arm_measures(records: list[dict], ground_truth: dict, payloads: dict) -> dict:
    """copy cells + M1-CO cells (+FP, paired discordant, d_i M2) for one arm's
    agent records."""
    cp = p3s.score_copy_records(records, ground_truth, payloads)
    rec = sc.score_records(records, ground_truth)
    return {
        "copy_cells": p3s.summarize_copy(cp),
        "measures": rpt.build_report(rec),
        "_cp": cp, "_rec": rec,  # kept for cross-section reuse; dropped before write
    }


def build_full_report(agent_records, transcription_records, ground_truth, payloads) -> dict:
    by_arm = {a: [r for r in agent_records if r.get("arm") == a] for a in ("A", "B", "C")}

    report: dict = {
        "n_agent_records": len(agent_records),
        "n_transcription_records": len(transcription_records),
        "n_by_arm": {a: len(by_arm[a]) for a in ("A", "B", "C")},
    }

    # ---- Arm A: primary + secondary (Sections 4, 5, 6) ----
    a = _arm_measures(by_arm["A"], ground_truth, payloads)
    cp_a, rec_a = a.pop("_cp"), a.pop("_rec")
    a["H1_P1_ko"] = _paired(cp_a, "P1", "ko")                 # Section 5, confirmatory
    a["P2_ko"] = _paired(cp_a, "P2", "ko")                    # Section 6, descriptive
    a["P1_en"] = _paired(cp_a, "P1", "en")
    a["P2_en"] = _paired(cp_a, "P2", "en")
    a["d_i_copy_P1"] = p3s.d_i_on_copy(cp_a, "P1")            # Section 6
    a["d_i_m2_P1"] = p3s.d_i_on_m2(rec_a, "P1")              # Section 6
    a["transcription_T"] = p3s.transcription_summary(
        [t for t in transcription_records if t.get("arm", "A") == "A"], payloads)
    report["arm_A"] = a

    # ---- Arm B: per font + agent-output determinism vs arm A (Section 7) ----
    arm_b = {}
    for font in ("NotoSansKR", "NanumGothic"):
        recs = [r for r in by_arm["B"] if r.get("font") == font]
        cp = p3s.score_copy_records(recs, ground_truth, payloads)
        arm_b[font] = {"copy_cells": p3s.summarize_copy(cp)}
    arm_b["determinism_P1_ko_NotoSansKR_vs_armA"] = _determinism_agent_outputs(by_arm["A"], by_arm["B"])
    arm_b["transcription_T"] = p3s.transcription_summary(
        [t for t in transcription_records if t.get("arm") == "B"], payloads)
    report["arm_B"] = arm_b

    # ---- Arm C: next to arm A (Section 8) ----
    c = _arm_measures(by_arm["C"], ground_truth, payloads)
    cp_c, rec_c = c.pop("_cp"), c.pop("_rec")
    c["d_i_copy_P1"] = p3s.d_i_on_copy(cp_c, "P1")
    c["d_i_m2_P1"] = p3s.d_i_on_m2(rec_c, "P1")
    report["arm_C"] = c

    # Footer line count per cell (Section 7 wrap effect: NotoSansKR x P1_en is 4
    # lines where arm A's P1_en is 3), reported next to the arm B results.
    report["footer_line_counts"] = _footer_line_counts(DATA_DIR, payloads)

    return report


def _determinism_agent_outputs(arm_a_records, arm_b_records) -> dict:
    """Section 7 determinism check (agent outputs): arm B P1_ko/NotoSansKR agent
    records should match arm A's P1_ko image agent records on raw_output,
    tool_calls and completion_sha256 for all present receipts (greedy decoding,
    identical model input). Reports how many match."""
    a_by_rid = {r["receipt_id"]: r for r in arm_a_records
                if r["variant"] == "P1_ko" and r["channel"] == "image"}
    b_by_rid = {r["receipt_id"]: r for r in arm_b_records
                if r["variant"] == "P1_ko" and r.get("font") == "NotoSansKR" and r["channel"] == "image"}
    common = sorted(set(a_by_rid) & set(b_by_rid))
    match = 0
    mismatches = []
    for rid in common:
        a, b = a_by_rid[rid], b_by_rid[rid]
        if all(a.get(f) == b.get(f) for f in ("raw_output", "tool_calls", "completion_sha256")):
            match += 1
        else:
            mismatches.append(rid)
    return {"n_compared": len(common), "n_match": match, "mismatch_receipt_ids": mismatches}


def write_reports(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase3b_results.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    L = []
    L.append("# Phase 3b R_eval results\n")
    L.append(f"Agent records: {report['n_agent_records']} "
             f"(A={report['n_by_arm']['A']}, B={report['n_by_arm']['B']}, C={report['n_by_arm']['C']}); "
             f"transcription records: {report['n_transcription_records']}\n")

    h1 = report["arm_A"]["H1_P1_ko"]
    nc = h1["newcombe"]
    L.append("## H1 (confirmatory): P1_ko copy, image > text (one-sided exact McNemar)\n")
    t = h1["table"]
    L.append(f"- discordant a (image1 text0) = {t['n10']}, b (image0 text1) = {t['n01']}, "
             f"n11={t['n11']} n00={t['n00']}; dropped={len(t['dropped_receipt_ids'])}")
    L.append(f"- one-sided p = {h1['mcnemar']['p_value_one_sided']:.6f}; "
             f"supported (alpha 0.05): {h1['mcnemar']['supported']}")
    if nc:
        L.append(f"- Newcombe 95% CI for image-text copy diff: [{nc['lower']:.4f}, {nc['upper']:.4f}] (d={nc['d']:.4f})")
    bs = h1["bootstrap"]
    if bs["ci_low"] is not None:
        L.append(f"- bootstrap (seed {bs['seed']}) 95% CI: [{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]\n")

    fcounts = report.get("footer_line_counts", {})

    def copy_table(title, cells, arm, font):
        L.append(f"\n## {title}\n")
        L.append("| variant|channel | footer_lines | n | copy | copy_rate | cond. copy | copy_first_call | wc_copy | copy/composed/other |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for key in sorted(cells):
            c = cells[key]
            cat = c["category"]
            variant = key.split("|")[0]
            fl = fcounts.get(f"{arm}|{font or '-'}|{variant}")
            L.append(f"| {key} | {fl} | {c['n']} | {c['copy']} | {c['copy_rate']} | {c['conditional_copy_rate']} | "
                     f"{c['copy_first_call']} | {c['worst_case_copy']} | "
                     f"{cat['copy']}/{cat['composed']}/{cat['other']} |")

    copy_table("Arm A copy (per variant|channel)", report["arm_A"]["copy_cells"], "A", None)
    for font in ("NotoSansKR", "NanumGothic"):
        copy_table(f"Arm B copy -- {font}", report["arm_B"][font]["copy_cells"], "B", font)
    copy_table("Arm C copy (per variant|channel)", report["arm_C"]["copy_cells"], "C", None)
    a_p1en = fcounts.get("A|-|P1_en")
    b_p1en = fcounts.get("B|NotoSansKR|P1_en")
    L.append(f"\n_Footer-line note: arm B NotoSansKR P1_en renders {b_p1en} footer lines vs "
             f"arm A P1_en's {a_p1en} (the Section 7 wrap effect; not a copy-rate effect)._")

    det = report["arm_B"]["determinism_P1_ko_NotoSansKR_vs_armA"]
    L.append(f"\n## Arm B determinism (P1_ko NotoSansKR agent outputs vs arm A): "
             f"{det['n_match']}/{det['n_compared']} match\n")

    di = report["arm_A"]
    L.append("## d_i (arm A), P1\n")
    L.append(f"- d_i on copy: counts {di['d_i_copy_P1']['counts']}, mean {di['d_i_copy_P1']['mean']}")
    L.append(f"- d_i on M2:   counts {di['d_i_m2_P1']['counts']}, mean {di['d_i_m2_P1']['mean']}\n")

    L.append("## T (transcription) exact-match / mean CER per cell (arm A)\n")
    L.append("| cell | n | exact_match | mean_cer |")
    L.append("|---|---|---|---|")
    for key in sorted(report["arm_A"]["transcription_T"]):
        tc = report["arm_A"]["transcription_T"][key]
        L.append(f"| {key} | {tc['n']} | {tc['exact_match']} | {tc['mean_cer']} |")

    md_path = out_dir / "phase3b_results.md"
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--receipts", type=Path, default=DATA_DIR / "receipts.jsonl")
    args = parser.parse_args()

    agent, transcription = load_records(args.input_dir)
    print(f"Loaded {len(agent)} agent records, {len(transcription)} transcription records")
    merge_check(agent, transcription, args.input_dir)  # refuses on bad count/dup/env
    print("Merge check OK")
    ground_truth = sc.load_ground_truth(args.receipts)
    payloads = sc.load_payloads()
    report = build_full_report(agent, transcription, ground_truth, payloads)
    jp, mp = write_reports(report, args.out_dir)
    print(f"Wrote {jp}, {mp}")


if __name__ == "__main__":
    main()
