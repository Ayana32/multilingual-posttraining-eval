"""`mpe-agent` command-line interface (docs/v1-spec.md Section 8).

Three subcommands:

    mpe-agent run    --model <id> --revision <40-hex> --base-url <url>
                     (--sample | --attack-set phase3b [--repo-root R]
                      | --manifest M --data-dir D --fixtures-dir F)
                     --channel image|text|both --out runs/<name>/
    mpe-agent score  runs/<name>/records.jsonl   # re-score, no re-running
    mpe-agent report runs/<name>/ --html report.html

Design rules carried from Step A + the three sign-off conditions:

1. Provenance (condition 1): every run records the path+sha256 of the
   imported harness/score/report modules, in run_metadata.json and on
   each record ("code_provenance"); a CI test checks byte-identity with
   phase3/ (mpe_agent.provenance).
2. No implicit defaults (condition 2): fixtures and ground truth are
   always resolved from an explicit source (--sample's bundled data, a
   --repo-root for --attack-set, or explicit --data-dir/--fixtures-dir/
   --ground-truth), NEVER from harness.__file__ or score.GROUND_TRUTH_PATH.
   The run fails fast if any manifest receipt_id is absent from the ground
   truth. The v1 manifest uses phase3b receipts, so the phase3b ground
   truth is phase3b/data/receipts.jsonl.
3. Sample labelling (condition 3): a --sample run stamps every record and
   the metadata with "sample": true, and the HTML report shows a banner;
   the numbers are a wiring check, never a finding.

Running is sequential (Section 5 fallback allows it); every record keeps
its retry_count, and run() logs retries and per-row status to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import scoring
from .openai_adapter import (
    CHANNELS,
    OpenAIAdapterConfig,
    load_fixtures,
    load_manifest,
    run_agent_openai,
)
from .provenance import module_provenance
from .report_html import build_html_report

_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"


# ---------------------------------------------------------------------------
# Source resolution (condition 2: explicit sources only, no module defaults)
# ---------------------------------------------------------------------------

class _CliError(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"mpe-agent: error: {message}")


def _resolve_run_sources(args: argparse.Namespace):
    """Returns (manifest_path, data_dir, fixtures_dir, ground_truth_path,
    repo_root, is_sample) from the mutually-exclusive source selectors,
    every path explicit."""
    explicit = any([args.manifest, args.data_dir, args.fixtures_dir])

    if args.sample:
        if explicit or args.attack_set:
            raise _CliError("--sample cannot be combined with --attack-set or explicit --manifest/--data-dir/--fixtures-dir")
        return (_SAMPLE_DIR / "manifest.jsonl", _SAMPLE_DIR,
                _SAMPLE_DIR / "fixtures", _SAMPLE_DIR / "receipts.jsonl",
                _SAMPLE_DIR, True)

    if args.attack_set:
        if args.attack_set != "phase3b":
            raise _CliError(f"unknown --attack-set {args.attack_set!r} (only 'phase3b')")
        root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
        data_dir = Path(args.data_dir).resolve() if args.data_dir else root / "phase3b" / "data"
        fixtures_dir = Path(args.fixtures_dir).resolve() if args.fixtures_dir else root / "phase3" / "fixtures"
        manifest = Path(args.manifest).resolve() if args.manifest else root / "runs" / "v1_manifest.jsonl"
        gt = Path(args.ground_truth).resolve() if args.ground_truth else data_dir / "receipts.jsonl"
        return (manifest, data_dir, fixtures_dir, gt, root, False)

    # Fully explicit.
    if not (args.manifest and args.data_dir and args.fixtures_dir):
        raise _CliError("choose a source: --sample, --attack-set phase3b, or all of --manifest/--data-dir/--fixtures-dir")
    data_dir = Path(args.data_dir).resolve()
    gt = Path(args.ground_truth).resolve() if args.ground_truth else data_dir / "receipts.jsonl"
    root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    return (Path(args.manifest).resolve(), data_dir,
            Path(args.fixtures_dir).resolve(), gt, root, False)


def _check_ground_truth_coverage(rows: list[dict], ground_truth: dict[str, dict]) -> None:
    """Condition 2: fail fast if any manifest receipt_id is missing from
    the ground truth, rather than discovering it mid-run (score_record
    would KeyError) or silently mis-scoring."""
    missing = sorted({r["receipt_id"] for r in rows} - set(ground_truth))
    if missing:
        raise _CliError(
            "ground truth is missing receipt_id(s) referenced by the manifest: "
            + ", ".join(missing)
            + ". The v1 manifest uses phase3b receipts, so --ground-truth / --data-dir "
            "must point at phase3b/data/receipts.jsonl."
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _channels_for(choice: str) -> tuple[str, ...]:
    if choice == "both":
        return CHANNELS
    if choice in CHANNELS:
        return (choice,)
    raise _CliError(f"unknown channel {choice!r}")


def cmd_run(args: argparse.Namespace) -> int:
    manifest_path, data_dir, fixtures_dir, gt_path, repo_root, is_sample = _resolve_run_sources(args)

    revision = args.revision
    if is_sample and not revision:
        revision = "0" * 40  # placeholder pinned-hash for the quickstart only

    for label, path in (("manifest", manifest_path), ("fixtures dir", fixtures_dir),
                        ("data dir", data_dir), ("ground truth", gt_path)):
        if not Path(path).exists():
            raise _CliError(f"{label} not found: {path}")

    rows = load_manifest(data_dir) if manifest_path == data_dir / "manifest.jsonl" \
        else [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    fixtures = load_fixtures(fixtures_dir)
    ground_truth = scoring.load_ground_truth(gt_path)
    _check_ground_truth_coverage(rows, ground_truth)

    channels = _channels_for(args.channel)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = out_dir.name
    planned = len(rows) * len(channels)
    provenance = module_provenance()

    config = OpenAIAdapterConfig(
        model_id=args.model,
        model_revision=revision,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        backend=args.backend,
        backend_version=args.backend_version,
        tool_choice=args.tool_choice,
        timeout=args.timeout,
        max_retries=args.max_retries,
        api_key=args.api_key,
    )

    records_path = out_dir / "records.jsonl"
    seq = 0
    total_retries = 0
    status_tally: dict[str, int] = {}
    print(f"[run] {planned} planned runs ({len(rows)} manifest rows x {len(channels)} channel(s)) -> {records_path}",
          file=sys.stderr)
    with records_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            for channel in channels:
                run_id = f"{run_name}/{seq:05d}"
                seq += 1
                record = run_agent_openai(row, channel, fixtures, config,
                                          data_dir=data_dir, repo_root=repo_root)
                record["run_id"] = run_id
                record["arm"] = row.get("arm")
                record["sample"] = is_sample
                record["code_provenance"] = provenance
                fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

                status = record["outcome_status"]
                status_tally[status] = status_tally.get(status, 0) + 1
                total_retries += record.get("retry_count", 0)
                retried = f" retries={record['retry_count']}" if record.get("retry_count") else ""
                print(f"[run] {run_id} {row['receipt_id']}/{row['variant']}/{channel} "
                      f"-> {status}{retried}", file=sys.stderr)

    metadata = {
        "run_name": run_name,
        "sample": is_sample,
        "planned_runs": planned,
        "observed_runs": seq,
        "channels": list(channels),
        "manifest_path": str(manifest_path),
        "manifest_rows": len(rows),
        "data_dir": str(data_dir),
        "fixtures_dir": str(fixtures_dir),
        "ground_truth_path": str(gt_path),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "base_url": config.base_url,
        "backend": config.backend,
        "backend_version": config.backend_version,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "total_retries": total_retries,
        "status_tally": status_tally,
        "code_provenance": provenance,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[run] done: {seq}/{planned} runs, statuses={status_tally}, total_retries={total_retries}",
          file=sys.stderr)
    print(f"[run] records: {records_path}", file=sys.stderr)
    print(f"[run] metadata: {out_dir / 'run_metadata.json'}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# score (re-score stored records without re-running)
# ---------------------------------------------------------------------------

def _ground_truth_for_records(args: argparse.Namespace, records_path: Path) -> Path:
    """Resolve the ground-truth path for score/report WITHOUT any module
    default (condition 2): an explicit --ground-truth, else the path the
    run recorded in run_metadata.json, else error."""
    if args.ground_truth:
        return Path(args.ground_truth).resolve()
    meta_path = records_path.parent / "run_metadata.json"
    if meta_path.exists():
        gt = json.loads(meta_path.read_text(encoding="utf-8")).get("ground_truth_path")
        if gt:
            return Path(gt)
    raise _CliError(
        "no ground truth: pass --ground-truth, or run from a directory whose "
        "run_metadata.json records ground_truth_path."
    )


def cmd_score(args: argparse.Namespace) -> int:
    records_path = Path(args.records).resolve()
    if not records_path.exists():
        raise _CliError(f"records not found: {records_path}")
    gt_path = _ground_truth_for_records(args, records_path)
    if not gt_path.exists():
        raise _CliError(f"ground truth not found: {gt_path}")

    records = scoring.load_jsonl(records_path)
    ground_truth = scoring.load_ground_truth(gt_path)
    _check_ground_truth_coverage(records, ground_truth)
    scores = scoring.score_records(records, ground_truth)

    out_path = Path(args.out).resolve() if args.out else records_path.parent / "scores.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for s in scores:
            fh.write(json.dumps(asdict(s), sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "cells": scoring.cells_with_wilson(scores),
        "run_status": scoring.run_status_counts(records),
        "n_records": len(records),
        "ground_truth_path": str(gt_path),
    }
    summary_path = records_path.parent / "scores_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"[score] {len(records)} records re-scored with {gt_path}", file=sys.stderr)
    print(f"[score] scores: {out_path}", file=sys.stderr)
    print(f"[score] summary: {summary_path}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# report (HTML)
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    records_path = run_dir if run_dir.is_file() else run_dir / "records.jsonl"
    if not records_path.exists():
        raise _CliError(f"records.jsonl not found in {run_dir}")
    base_dir = records_path.parent
    gt_path = _ground_truth_for_records(args, records_path)
    if not gt_path.exists():
        raise _CliError(f"ground truth not found: {gt_path}")

    records = scoring.load_jsonl(records_path)
    ground_truth = scoring.load_ground_truth(gt_path)
    _check_ground_truth_coverage(records, ground_truth)

    meta_path = base_dir / "run_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    planned = args.planned if args.planned is not None else metadata.get("planned_runs")

    html = build_html_report(records, ground_truth, metadata=metadata, planned_runs=planned)
    html_path = Path(args.html).resolve() if args.html else base_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[report] wrote {html_path}", file=sys.stderr)
    print(str(html_path))  # stdout: the report path, for scripting
    return 0


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mpe-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the harness against an OpenAI-compatible endpoint")
    r.add_argument("--model", required=True, help="model id sent in the request")
    r.add_argument("--revision", help="pinned 40-hex commit revision (required unless --sample)")
    r.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (…/v1)")
    r.add_argument("--channel", default="both", choices=("image", "text", "both"))
    r.add_argument("--out", required=True, help="output run directory")
    # source selectors
    r.add_argument("--sample", action="store_true", help="use the bundled quickstart sample fixture")
    r.add_argument("--attack-set", help="named attack set (phase3b)")
    r.add_argument("--repo-root", help="repo root for --attack-set path resolution (default: CWD)")
    r.add_argument("--manifest", help="explicit manifest .jsonl")
    r.add_argument("--data-dir", help="explicit data dir (images/, transcripts/, receipts.jsonl)")
    r.add_argument("--fixtures-dir", help="explicit fixtures dir (system_prompt/tools/prompts)")
    r.add_argument("--ground-truth", help="explicit ground-truth receipts.jsonl (default: <data-dir>/receipts.jsonl)")
    # endpoint config
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--backend", default="openai-compatible")
    r.add_argument("--backend-version", default=None)
    r.add_argument("--tool-choice", default="auto")
    r.add_argument("--timeout", type=float, default=60.0)
    r.add_argument("--max-retries", type=int, default=2)
    r.add_argument("--api-key", default=None)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="re-score a records.jsonl without re-running")
    s.add_argument("records", help="path to records.jsonl")
    s.add_argument("--ground-truth", help="ground-truth receipts.jsonl (else read from run_metadata.json)")
    s.add_argument("--out", help="scores output path (default: <run>/scores.jsonl)")
    s.set_defaults(func=cmd_score)

    rep = sub.add_parser("report", help="build the HTML report for a run directory")
    rep.add_argument("run_dir", help="run directory (or a records.jsonl path)")
    rep.add_argument("--html", help="output HTML path (default: <run>/report.html)")
    rep.add_argument("--ground-truth", help="ground-truth receipts.jsonl (else read from run_metadata.json)")
    rep.add_argument("--planned", type=int, default=None, help="planned-run total (else from run_metadata.json)")
    rep.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
