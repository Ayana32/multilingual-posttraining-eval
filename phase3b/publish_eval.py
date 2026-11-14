#!/usr/bin/env python3
"""Phase 3b publish step: copy the R_eval chunk outputs from the gitignored
pod results into the committed report/phase3b_eval/, with integrity guards.

For every chunk in phase3b/run.py's fixed plan, exactly one result directory
must be successful -- exit code 0 in the chunk status log AND a record count
equal to the plan's expected count for that chunk. This refuses on:
  - missing: no successful directory for a chunk;
  - short:   a successful (exit-0) directory with fewer records than planned;
  - over:    more records than planned;
  - duplicate: more than one successful directory for the same chunk (e.g. both
    the original and a retry left exit-0 and full).
Retries are matched by a `<chunk>_retry[N]` directory name.

Copies each chunk's record file and environment.json into
report/phase3b_eval/<chunk>/ and writes report/phase3b_eval/SHA256SUMS, which
phase3b/score_eval.py verifies before reading anything. Run on the Mac after the
pod results are copied back and verified; commits happen there.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import harness as hz  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load("phase3b_run", "phase3b/run.py")

DATA_DIR = REPO_ROOT / "phase3b" / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "phase3b" / "eval"
DEFAULT_OUT_DIR = REPO_ROOT / "report" / "phase3b_eval"
RECORD_FILE = {"agent": "records.jsonl", "transcription": "transcription_records.jsonl"}


class PublishError(RuntimeError):
    """Raised when a chunk is missing, short, over-long or duplicated."""


def parse_status_log(path: Path) -> dict[str, int]:
    """Parses `<name> exit_code=<n>` lines (run_all_chunks.sh's status log)."""
    status: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+)\s+exit_code=(-?\d+)\s*$", line.strip())
        if m:
            status[m.group(1)] = int(m.group(2))
    return status


def build_plan(manifest: list[dict], chunks: list[dict] | None = None) -> list[dict]:
    """Per-chunk expected record count from run.tasks_for_chunk."""
    chunks = chunks if chunks is not None else run.chunk_plan()
    return [
        {"name": c["name"], "kind": c["kind"], "expected_count": len(run.tasks_for_chunk(manifest, c))}
        for c in chunks
    ]


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _candidate_dirs(results_dir: Path, name: str) -> list[Path]:
    pat = re.compile(rf"^{re.escape(name)}(_retry\d*)?$")
    return sorted(d for d in results_dir.iterdir() if d.is_dir() and pat.match(d.name))


def select_chunk_dir(results_dir: Path, item: dict, status: dict[str, int]) -> Path:
    name, kind, expected = item["name"], item["kind"], item["expected_count"]
    rf = RECORD_FILE[kind]
    cands = _candidate_dirs(results_dir, name)
    if not cands:
        raise PublishError(f"missing chunk {name!r}: no result directory under {results_dir}")

    exit0 = [d for d in cands if status.get(d.name) == 0 and (d / rf).exists()]
    for d in exit0:
        c = _count(d / rf)
        if c < expected:
            raise PublishError(f"short chunk {d.name!r}: {c} records < planned {expected}")
        if c > expected:
            raise PublishError(f"over-long chunk {d.name!r}: {c} records > planned {expected}")
    good = [d for d in exit0 if _count(d / rf) == expected]
    if not good:
        raise PublishError(
            f"missing chunk {name!r}: no exit-0 directory with the planned {expected} records "
            f"(candidates: {[d.name for d in cands]})")
    if len(good) > 1:
        raise PublishError(f"duplicate chunk {name!r}: more than one successful directory {[d.name for d in good]}")
    return good[0]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256sums(out_dir: Path) -> Path:
    lines = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{sha256_file(p)}  {p.relative_to(out_dir)}")
    sums = out_dir / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums


def publish(results_dir: Path, out_dir: Path, status: dict[str, int], plan: list[dict]) -> list[Path]:
    """Validates and copies every chunk; writes SHA256SUMS. Raises PublishError
    (before writing SHA256SUMS) on any missing/short/over/duplicate chunk."""
    selected = {item["name"]: select_chunk_dir(results_dir, item, status) for item in plan}  # validates all first
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for item in plan:
        src = selected[item["name"]]
        dst = out_dir / item["name"]
        dst.mkdir(parents=True, exist_ok=True)
        rf = RECORD_FILE[item["kind"]]
        shutil.copyfile(src / rf, dst / rf)
        copied.append(dst / rf)
        env = src / "environment.json"
        if env.exists():
            shutil.copyfile(env, dst / "environment.json")
            copied.append(dst / "environment.json")
    write_sha256sums(out_dir)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--status-log", type=Path, required=True,
                        help="the chunk status log (run_all_chunks.sh's phase3b_chunk_status.log)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    status = parse_status_log(args.status_log)
    manifest = hz.load_manifest(args.data_dir)
    plan = build_plan(manifest)
    copied = publish(args.results_dir, args.out_dir, status, plan)
    print(f"Published {len(plan)} chunks ({len(copied)} files) to {args.out_dir}; wrote SHA256SUMS")


if __name__ == "__main__":
    main()
