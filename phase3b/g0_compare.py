#!/usr/bin/env python3
"""Phase 3b G0 additional check (docs/phase3b-replication.md Section 10, "G0,
additionally"): re-run Phase 3's own 20 P1_ko image eval runs and diff them,
byte for byte, against the committed Phase 3 records in report/phase3_eval/.

This module is the COMPARISON, not the re-run: running the model is an
evaluation run and is out of scope here. It takes a re-run output directory
(gitignored, e.g. results/phase3b/g0_rerun_check/, produced later on the pod)
and the committed report/phase3_eval/, and produces the summary that Section 10
requires to be committed (report/phase3b/g0_rerun_summary.{json,md}).

Mismatch rule, ordered exactly as Section 10 states it:
  1. Compare the re-run's environment.json against the committed one on the
     version keys (transformers/torch/cuda/gpu/pillow/python; git_commit and
     git_dirty are expected to differ and are excluded). If any version
     differs, THAT is the cause to fix first -- align the environment and
     re-run before reading anything into the output diff.
  2. If versions match and outputs still differ, it is a real
     model/harness nondeterminism finding: it does not block Phase 3b (log it in
     docs/phase3b-deviations.md and proceed), and the re-run copy count over the
     20 P1_ko image records is reported next to Phase 3's own 16.
  3. The summary (per-receipt match/mismatch, both version sets, the re-run copy
     count) is committed; the raw re-run records are not.

report/phase3_eval/SHA256SUMS is verified before any record is read, and the
same check is meant to be run again after the re-run (verify_committed_intact).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import score as sc  # noqa: E402
import exploratory_m3 as em  # noqa: E402  (verify_sha256sums, reused)

EVAL_DIR = REPO_ROOT / "report" / "phase3_eval"
AGENT_CHUNKS = [f"chunk_{i:02d}" for i in range(1, 9)]

# git_commit/git_dirty are expected to differ (Section 10) and are not part of
# this version check.
VERSION_KEYS = [
    "transformers_version", "torch_version", "cuda_version",
    "gpu_name", "pillow_version", "python_version",
]
COMPARE_FIELDS = ["rendered_prompts", "raw_output", "tool_calls", "final_text", "completion_sha256"]

PHASE3_P1_KO_IMAGE_COPY = 16  # docs/phase3-results.md Section 5a


def _load_p3b_score():
    spec = importlib.util.spec_from_file_location("phase3b_score", REPO_ROOT / "phase3b" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase3b_score"] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_committed_intact(eval_dir: Path = EVAL_DIR) -> None:
    """Re-verify report/phase3_eval/SHA256SUMS (the frozen Phase 3 records this
    re-run is diffed against must be intact, before AND after)."""
    em.verify_sha256sums(eval_dir)


def load_committed_p1ko_image(eval_dir: Path = EVAL_DIR) -> dict[str, dict]:
    verify_committed_intact(eval_dir)
    out: dict[str, dict] = {}
    for chunk in AGENT_CHUNKS:
        path = eval_dir / chunk / "records.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["variant"] == "P1_ko" and r["channel"] == "image":
                out[r["receipt_id"]] = r
    return out


def _env_versions(env: dict) -> dict:
    return {k: env.get(k) for k in VERSION_KEYS}


def committed_env_versions(eval_dir: Path = EVAL_DIR) -> dict:
    env = json.loads((eval_dir / "chunk_01" / "environment.json").read_text(encoding="utf-8"))
    return _env_versions(env)


def load_rerun(rerun_dir: Path) -> tuple[dict[str, dict], dict]:
    records = {}
    for line in (rerun_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["variant"] == "P1_ko" and r["channel"] == "image":
                records[r["receipt_id"]] = r
    env = json.loads((rerun_dir / "environment.json").read_text(encoding="utf-8"))
    return records, _env_versions(env)


def compare_env(committed: dict, rerun: dict) -> dict:
    diffs = {k: {"committed": committed.get(k), "rerun": rerun.get(k)}
             for k in VERSION_KEYS if committed.get(k) != rerun.get(k)}
    return {"match": not diffs, "committed": committed, "rerun": rerun, "diffs": diffs}


def compare_records(committed: dict[str, dict], rerun: dict[str, dict]) -> dict:
    per_receipt = []
    n_match = 0
    for rid in sorted(committed):
        c = committed[rid]
        r = rerun.get(rid)
        if r is None:
            per_receipt.append({"receipt_id": rid, "match": False, "differing_fields": ["<missing in re-run>"]})
            continue
        differing = [f for f in COMPARE_FIELDS if c.get(f) != r.get(f)]
        match = not differing
        n_match += int(match)
        per_receipt.append({"receipt_id": rid, "match": match, "differing_fields": differing})
    return {"n_committed": len(committed), "n_match": n_match,
            "n_mismatch": len(committed) - n_match, "per_receipt": per_receipt}


def rerun_copy_count(rerun: dict[str, dict]) -> int:
    p3s = _load_p3b_score()
    gt = sc.load_ground_truth()
    payloads = sc.load_payloads()
    payload_text = payloads["P1"]["ko"]
    count = 0
    for rid, rec in rerun.items():
        g = gt[rid]
        count += p3s.copy_value(rec, g["vendor"], g["total_amount"], payload_text)
    return count


def build_summary(rerun_dir: Path, eval_dir: Path = EVAL_DIR) -> dict:
    committed = load_committed_p1ko_image(eval_dir)
    committed_env = committed_env_versions(eval_dir)
    rerun, rerun_env = load_rerun(rerun_dir)

    env_cmp = compare_env(committed_env, rerun_env)
    rec_cmp = compare_records(committed, rerun)

    if not env_cmp["match"]:
        verdict = "version_mismatch_fix_first"
        copy_count = None
    elif rec_cmp["n_mismatch"] > 0:
        verdict = "nondeterminism_finding"
        copy_count = rerun_copy_count(rerun)
    else:
        verdict = "match"
        copy_count = rerun_copy_count(rerun)

    return {
        "check": "G0 additional: re-run Phase 3 P1_ko image vs committed records, byte for byte",
        "verdict": verdict,
        "n_committed": rec_cmp["n_committed"],
        "n_match": rec_cmp["n_match"],
        "n_mismatch": rec_cmp["n_mismatch"],
        "environment": env_cmp,
        "rerun_copy_count": copy_count,
        "phase3_copy_count": PHASE3_P1_KO_IMAGE_COPY,
        "compare_fields": COMPARE_FIELDS,
        "per_receipt": rec_cmp["per_receipt"],
    }


def write_summary(summary: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "g0_rerun_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Phase 3b G0 re-run comparison summary", "",
        f"**Verdict:** `{summary['verdict']}`", "",
        f"- records compared (P1_ko image): {summary['n_match']}/{summary['n_committed']} byte-identical "
        f"({summary['n_mismatch']} differ)",
        f"- re-run copy count: {summary['rerun_copy_count']} (Phase 3: {summary['phase3_copy_count']})",
        f"- environment versions match: {summary['environment']['match']}",
        "",
        "## Environment versions", "",
        "| key | committed | re-run |", "|---|---|---|",
    ]
    for k in VERSION_KEYS:
        lines.append(f"| {k} | {summary['environment']['committed'].get(k)} | {summary['environment']['rerun'].get(k)} |")
    lines += ["", "## Per-receipt", "", "| receipt_id | match | differing fields |", "|---|---|---|"]
    for row in summary["per_receipt"]:
        lines.append(f"| {row['receipt_id']} | {row['match']} | {', '.join(row['differing_fields']) or '-'} |")
    md_path = out_dir / "g0_rerun_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-dir", type=Path, required=True,
                        help="gitignored dir with the re-run's records.jsonl + environment.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "report" / "phase3b")
    args = parser.parse_args()
    summary = build_summary(args.rerun_dir)
    jp, mp = write_summary(summary, args.out_dir)
    verify_committed_intact()  # the committed records must still be intact afterwards
    print(f"G0 re-run: verdict={summary['verdict']} match={summary['n_match']}/{summary['n_committed']} "
          f"-> {jp}, {mp}")


if __name__ == "__main__":
    main()
