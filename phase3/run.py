#!/usr/bin/env python3
"""Phase 3 run CLI: runs selected phase3/data/manifest.jsonl rows and
channels through phase3/harness.py's run_agent, and the separate
transcription calls (docs/phase3-visual-injection.md Section 4.4),
writing records.jsonl (or transcription_records.jsonl) plus a run-level
environment.json to an output directory under results/ (gitignored).

Config (model_id, model_revision, max_new_tokens, device, dtype,
commit_b_tag) comes from phase3/config.yaml by default; model_revision is
a placeholder there until Commit B pins a real one (see that file and
docs/phase3-visual-injection.md Section 6.3).

Every row goes through harness.py's guards: an eval-set row is refused
unless commit_b_tag is set and the git tag phase3-prereg-B exists, and a
non-commit-hash model_revision is refused outright (see
phase3/harness.py's guard_eval_run / validate_model_revision). A
harness_error run (agent or transcription) is re-run once; the re-run
record is the one written out, even if it fails again (see
docs/phase3-deviations.md's harness_error re-run and exclusion rule --
phase3/score.py is what actually excludes a twice-failed run).

This module never runs a real model on import; build_hf_template_fn and
build_hf_generate_fn (phase3/harness.py) are only invoked from main().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import harness as hz  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "phase3" / "config.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "phase3"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> hz.HarnessConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return hz.HarnessConfig(
        model_id=data["model_id"],
        model_revision=data["model_revision"],
        max_new_tokens=data["max_new_tokens"],
        commit_b_tag=data.get("commit_b_tag", False),
        device=data.get("device", "cpu"),
        dtype=data.get("dtype", "auto"),
    )


def select_rows(
    manifest: list[dict],
    receipt_set: str,
    variants: set[str] | None = None,
    receipt_ids: set[str] | None = None,
) -> list[dict]:
    rows = [r for r in manifest if r["receipt_set"] == receipt_set]
    if variants is not None:
        rows = [r for r in rows if r["variant"] in variants]
    if receipt_ids is not None:
        rows = [r for r in rows if r["receipt_id"] in receipt_ids]
    return rows


# ---------------------------------------------------------------------------
# Agent runs (docs/phase3-visual-injection.md Section 4.4, "Runs")
# ---------------------------------------------------------------------------

def run_agent_rows(
    rows_and_channels: list[tuple[dict, str]],
    fixtures: hz.Fixtures,
    template_fn,
    generate_fn,
    config: hz.HarnessConfig,
    data_dir: Path = hz.DATA_DIR,
    repo_root: Path = hz.REPO_ROOT,
) -> list[dict]:
    """Runs run_agent for each (row, channel) pair, in order, re-running
    once on harness_error and keeping whatever the re-run produces (even
    if it fails again -- phase3/score.py is what excludes a record whose
    final outcome_status is still harness_error)."""
    records = []
    for row, channel in rows_and_channels:
        record = hz.run_agent(row, channel, fixtures, template_fn, generate_fn, config, data_dir, repo_root)
        if record["outcome_status"] == "harness_error":
            record = hz.run_agent(row, channel, fixtures, template_fn, generate_fn, config, data_dir, repo_root)
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Transcription runs (Section 4.4): a separate, single-turn, tool-free
# call per image that has an extra line, asking the model to transcribe
# everything in the image. Image channel only -- there is nothing to
# transcribe in the text channel.
# ---------------------------------------------------------------------------

def build_transcription_input(row: dict, fixtures: hz.Fixtures, data_dir: Path = hz.DATA_DIR) -> dict:
    image_block = {"type": "image", "image": hz.load_image(data_dir, row["image_path"])}
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": fixtures.transcription_prompt + "\n"},
            image_block,
        ]},
    ]
    return {"messages": messages, "tools": []}


def run_transcription(
    row: dict,
    fixtures: hz.Fixtures,
    template_fn,
    generate_fn,
    config: hz.HarnessConfig,
    data_dir: Path = hz.DATA_DIR,
    repo_root: Path = hz.REPO_ROOT,
) -> dict:
    """Runs one transcription call for an image-channel row that has an
    extra line (row["payload_id"] is not None). Not a tool-calling run --
    no outcome_status/tool_calls; the raw transcribed text is scored
    separately (T, Section 7) against the ground-truth payload string."""
    if row["payload_id"] is None:
        raise ValueError(f"row {row['receipt_id']}/{row['variant']} has no extra line to transcribe")

    hz.validate_model_revision(config.model_revision)
    hz.guard_eval_run(row, config.commit_b_tag, repo_root)

    model_input = build_transcription_input(row, fixtures, data_dir)
    rendered_prompt = template_fn(model_input["messages"], model_input["tools"])
    prompt_sha256 = hz.prompt_sha256_for_rendered(rendered_prompt, row["image_sha256"])
    images = [
        block["image"]
        for block in model_input["messages"][0]["content"]
        if block.get("type") == "image"
    ]

    raw_output, hit_max_new_tokens = generate_fn(rendered_prompt, images)

    return {
        "receipt_id": row["receipt_id"],
        "receipt_set": row["receipt_set"],
        "variant": row["variant"],
        "payload_id": row["payload_id"],
        "payload_lang": row["payload_lang"],
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "rendered_prompt": rendered_prompt,
        "prompt_sha256": prompt_sha256,
        "raw_output": raw_output,
        "hit_max_new_tokens": hit_max_new_tokens,
    }


def run_transcription_rows(
    rows: list[dict],
    fixtures: hz.Fixtures,
    template_fn,
    generate_fn,
    config: hz.HarnessConfig,
    data_dir: Path = hz.DATA_DIR,
    repo_root: Path = hz.REPO_ROOT,
) -> list[dict]:
    return [
        run_transcription(row, fixtures, template_fn, generate_fn, config, data_dir, repo_root)
        for row in rows
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_set_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--receipt-set", choices=["tune", "eval"], default="tune")
    parser.add_argument("--channels", default="image,text", help="comma-separated: image,text")
    parser.add_argument("--variants", default=None, help="comma-separated variant filter, default: all")
    parser.add_argument("--receipt-ids", default=None, help="comma-separated receipt_id filter, default: all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=hz.DATA_DIR)
    parser.add_argument(
        "--transcription", action="store_true",
        help="run the separate transcription calls instead of the agent (Section 4.4)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    fixtures = hz.load_fixtures()
    manifest = hz.load_manifest(args.data_dir)
    rows = select_rows(
        manifest, args.receipt_set,
        variants=_parse_set_arg(args.variants),
        receipt_ids=_parse_set_arg(args.receipt_ids),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hz.start_run(config, args.out_dir)

    template_fn = hz.build_hf_template_fn(config)
    generate_fn = hz.build_hf_generate_fn(config)

    if args.transcription:
        rows = [r for r in rows if r["payload_id"] is not None]
        records = run_transcription_rows(rows, fixtures, template_fn, generate_fn, config, args.data_dir)
        out_path = args.out_dir / "transcription_records.jsonl"
    else:
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
        rows_and_channels = [(row, channel) for row in rows for channel in channels]
        records = run_agent_rows(rows_and_channels, fixtures, template_fn, generate_fn, config, args.data_dir)
        out_path = args.out_dir / "records.jsonl"

    hz.write_records(records, out_path)
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
