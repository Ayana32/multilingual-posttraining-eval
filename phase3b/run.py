#!/usr/bin/env python3
"""Phase 3b run CLI (docs/phase3b-replication.md Section 9's run table), new
code that reuses phase3/harness.py and phase3/run.py unchanged.

Arms and runs (Section 9):
  - arm A (undefended): 7 variants x 2 channels x 40 = 560 agent runs.
  - arm B (font sensitivity): P1_en, P1_ko x image x 40 x 2 fonts = 160 agent
    runs (uses arm A's system prompt; only the footer font differs).
  - arm C (defence): clean, P1_en, P1_ko x 2 channels x 40 = 240 agent runs,
    run with arm_c_fixtures (arm A's images/transcripts, arm C's system prompt).
  - transcription: arm A 6 payload-bearing variants x 40 = 240; arm B P1_en,
    P1_ko x 40 x 2 fonts = 160; arm C reuses arm A's (0 new).
Totals: 960 agent runs, 400 transcription runs.

Every row is receipt_set == "phase3b_eval", so every agent run goes through
harness.run_agent's guard_phase3b_eval_run (tag + HEAD-descendant) unchanged;
transcription runs apply the same guard here. harness_error agent runs are
re-run once (the re-run record is kept even if it fails again; phase3b/score.py
/ phase3/score.py exclude a twice-failed run). Each record is annotated with its
arm and font. Records keep phase3/harness.py's format otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import harness as hz  # noqa: E402
import run as p3run  # noqa: E402  (phase3/run.py: load_config, build_transcription_input)

# phase3b.arm_c, loaded by path (arm_c imports harness; unique module name here).
import importlib.util  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arm_c = _load("phase3b_arm_c", "phase3b/arm_c.py")

DATA_DIR = REPO_ROOT / "phase3b" / "data"
DEFAULT_CONFIG_PATH = REPO_ROOT / "phase3" / "config.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "phase3b"

ARM_A_VARIANTS = ["clean", "neutral_en", "neutral_ko", "P1_en", "P1_ko", "P2_en", "P2_ko"]
ARM_B_VARIANTS = ["P1_en", "P1_ko"]
ARM_B_FONTS = ["NotoSansKR", "NanumGothic"]
ARM_C_VARIANTS = ["clean", "P1_en", "P1_ko"]
# Arm A transcription: the 6 payload-bearing variants (clean has no extra line).
ARM_A_TRANSCRIPTION_VARIANTS = ["neutral_en", "neutral_ko", "P1_en", "P1_ko", "P2_en", "P2_ko"]
CHANNELS = ["image", "text"]
RECEIPT_IDS = [f"phase3b_eval_{i:02d}" for i in range(1, 41)]


# ---------------------------------------------------------------------------
# Task building (a task = one record to produce)
# ---------------------------------------------------------------------------

def _arm_a_rows(manifest: list[dict]) -> list[dict]:
    return [r for r in manifest if r.get("arm") == "A"]


def _arm_b_rows(manifest: list[dict]) -> list[dict]:
    return [r for r in manifest if r.get("arm") == "B"]


def build_agent_tasks(manifest: list[dict]) -> list[dict]:
    """The full 960 agent tasks (560 A + 160 B + 240 C)."""
    tasks: list[dict] = []
    a_rows = {(r["receipt_id"], r["variant"]): r for r in _arm_a_rows(manifest)}

    # arm A: 7 variants x 2 channels x 40
    for (rid, variant), row in a_rows.items():
        if variant in ARM_A_VARIANTS:
            for channel in CHANNELS:
                tasks.append({"arm": "A", "font": None, "channel": channel, "kind": "agent", "row": row})
    # arm B: P1_en, P1_ko x image x 40 x 2 fonts
    for row in _arm_b_rows(manifest):
        if row["variant"] in ARM_B_VARIANTS:
            tasks.append({"arm": "B", "font": row.get("font"), "channel": "image", "kind": "agent", "row": row})
    # arm C: clean, P1_en, P1_ko x 2 channels x 40 (arm A rows, arm C prompt)
    for (rid, variant), row in a_rows.items():
        if variant in ARM_C_VARIANTS:
            for channel in CHANNELS:
                tasks.append({"arm": "C", "font": None, "channel": channel, "kind": "agent", "row": row})
    return tasks


def build_transcription_tasks(manifest: list[dict]) -> list[dict]:
    """The full 400 transcription tasks (240 A + 160 B)."""
    tasks: list[dict] = []
    for row in _arm_a_rows(manifest):
        if row["variant"] in ARM_A_TRANSCRIPTION_VARIANTS:
            tasks.append({"arm": "A", "font": None, "channel": "image", "kind": "transcription", "row": row})
    for row in _arm_b_rows(manifest):
        if row["variant"] in ARM_B_VARIANTS:
            tasks.append({"arm": "B", "font": row.get("font"), "channel": "image", "kind": "transcription", "row": row})
    return tasks


def task_key(task: dict) -> tuple:
    """A unique identity for a task (used to check the chunk plan partitions
    the full task set with no overlap)."""
    r = task["row"]
    return (task["kind"], task["arm"], task["font"], task["channel"], r["receipt_id"], r["variant"])


# ---------------------------------------------------------------------------
# Fixed chunk plan (Section 9; same per-chunk / status-log pattern as Phase 3)
# ---------------------------------------------------------------------------

_GROUPS = {
    "g1": RECEIPT_IDS[0:10], "g2": RECEIPT_IDS[10:20],
    "g3": RECEIPT_IDS[20:30], "g4": RECEIPT_IDS[30:40],
}
_HALVES = {"h1": RECEIPT_IDS[0:20], "h2": RECEIPT_IDS[20:40]}


def chunk_plan() -> list[dict]:
    """A fixed partition of all 960 agent + 400 transcription tasks into chunks,
    each a separate run.py invocation with its own --out-dir (so losing one
    chunk to a crash is cheap to redo, same approach as Phase 3's R_eval)."""
    chunks: list[dict] = []
    # Arm A agent: channel x receipt-group -> 7 variants x 10 = 70 each (8 chunks).
    for channel in CHANNELS:
        for gname, rids in _GROUPS.items():
            chunks.append({"name": f"agent_A_{channel}_{gname}", "kind": "agent", "arm": "A",
                           "channels": [channel], "variants": ARM_A_VARIANTS, "fonts": None, "receipt_ids": rids})
    # Arm C agent: channel x receipt-group -> 3 variants x 10 = 30 each (8 chunks).
    for channel in CHANNELS:
        for gname, rids in _GROUPS.items():
            chunks.append({"name": f"agent_C_{channel}_{gname}", "kind": "agent", "arm": "C",
                           "channels": [channel], "variants": ARM_C_VARIANTS, "fonts": None, "receipt_ids": rids})
    # Arm B agent: font x receipt-half -> 2 variants x 20 = 40 each (4 chunks).
    for font in ARM_B_FONTS:
        for hname, rids in _HALVES.items():
            chunks.append({"name": f"agent_B_{font}_{hname}", "kind": "agent", "arm": "B",
                           "channels": ["image"], "variants": ARM_B_VARIANTS, "fonts": [font], "receipt_ids": rids})
    # Arm A transcription: receipt-group -> 6 variants x 10 = 60 each (4 chunks).
    for gname, rids in _GROUPS.items():
        chunks.append({"name": f"transcription_A_{gname}", "kind": "transcription", "arm": "A",
                       "channels": ["image"], "variants": ARM_A_TRANSCRIPTION_VARIANTS, "fonts": None, "receipt_ids": rids})
    # Arm B transcription: font x receipt-half -> 2 variants x 20 = 40 each (4 chunks).
    for font in ARM_B_FONTS:
        for hname, rids in _HALVES.items():
            chunks.append({"name": f"transcription_B_{font}_{hname}", "kind": "transcription", "arm": "B",
                           "channels": ["image"], "variants": ARM_B_VARIANTS, "fonts": [font], "receipt_ids": rids})
    return chunks


def tasks_for_chunk(manifest: list[dict], chunk: dict) -> list[dict]:
    rids = set(chunk["receipt_ids"])
    variants = set(chunk["variants"])
    fonts = set(chunk["fonts"]) if chunk["fonts"] else None

    if chunk["arm"] == "B":
        base = [r for r in _arm_b_rows(manifest)
                if r["variant"] in variants and r["receipt_id"] in rids
                and (fonts is None or r.get("font") in fonts)]
    else:  # arm A and arm C both draw from arm A rows
        base = [r for r in _arm_a_rows(manifest)
                if r["variant"] in variants and r["receipt_id"] in rids]

    tasks: list[dict] = []
    for row in base:
        font = row.get("font") if chunk["arm"] == "B" else None
        if chunk["kind"] == "agent":
            for channel in chunk["channels"]:
                tasks.append({"arm": chunk["arm"], "font": font, "channel": channel, "kind": "agent", "row": row})
        else:
            tasks.append({"arm": chunk["arm"], "font": font, "channel": "image", "kind": "transcription", "row": row})
    return tasks


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _fixtures_for(arm: str, base: hz.Fixtures, arm_c_fx: hz.Fixtures) -> hz.Fixtures:
    return arm_c_fx if arm == "C" else base


def run_agent_task(task, base_fx, arm_c_fx, template_fn, generate_fn, config,
                   data_dir=DATA_DIR, repo_root=hz.REPO_ROOT) -> dict:
    fixtures = _fixtures_for(task["arm"], base_fx, arm_c_fx)
    row, channel = task["row"], task["channel"]
    rec = hz.run_agent(row, channel, fixtures, template_fn, generate_fn, config, data_dir, repo_root)
    if rec["outcome_status"] == "harness_error":
        rec = hz.run_agent(row, channel, fixtures, template_fn, generate_fn, config, data_dir, repo_root)
    rec["arm"] = task["arm"]
    rec["font"] = task["font"]
    return rec


def run_transcription_task(task, base_fx, arm_c_fx, template_fn, generate_fn, config,
                           data_dir=DATA_DIR, repo_root=hz.REPO_ROOT) -> dict:
    """A transcription run for a phase3b_eval row, with the phase3b eval guard
    applied here (phase3/run.py's run_transcription only applies Phase 3's
    guard, which early-returns for a phase3b_eval row). Otherwise identical to
    phase3/run.py's transcription logic, reused via build_transcription_input."""
    row = task["row"]
    fixtures = _fixtures_for(task["arm"], base_fx, arm_c_fx)
    hz.validate_model_revision(config.model_revision)
    hz.guard_phase3b_eval_run(row, repo_root)

    model_input = p3run.build_transcription_input(row, fixtures, data_dir)
    rendered_prompt = template_fn(model_input["messages"], model_input["tools"])
    prompt_sha256 = hz.prompt_sha256_for_rendered(rendered_prompt, row["image_sha256"])
    images = [b["image"] for b in model_input["messages"][0]["content"] if b.get("type") == "image"]
    raw_output, hit_max = generate_fn(rendered_prompt, images)
    return {
        "receipt_id": row["receipt_id"], "receipt_set": row["receipt_set"],
        "variant": row["variant"], "payload_id": row["payload_id"], "payload_lang": row["payload_lang"],
        "arm": task["arm"], "font": task["font"],
        "model_id": config.model_id, "model_revision": config.model_revision,
        "rendered_prompt": rendered_prompt, "prompt_sha256": prompt_sha256,
        "raw_output": raw_output, "hit_max_new_tokens": hit_max,
    }


def run_tasks(tasks, base_fx, arm_c_fx, template_fn, generate_fn, config,
              data_dir=DATA_DIR, repo_root=hz.REPO_ROOT) -> list[dict]:
    records = []
    for task in tasks:
        if task["kind"] == "agent":
            records.append(run_agent_task(task, base_fx, arm_c_fx, template_fn, generate_fn, config, data_dir, repo_root))
        else:
            records.append(run_transcription_task(task, base_fx, arm_c_fx, template_fn, generate_fn, config, data_dir, repo_root))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--chunk", help="chunk name from chunk_plan()")
    parser.add_argument("--list-chunks", action="store_true", help="print chunk names (for the launcher) and exit")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if args.list_chunks:
        for c in chunk_plan():
            print(c["name"])
        return
    if not args.chunk:
        raise SystemExit("--chunk is required (or use --list-chunks)")

    chunks = {c["name"]: c for c in chunk_plan()}
    if args.chunk not in chunks:
        raise SystemExit(f"unknown chunk {args.chunk!r}; known: {sorted(chunks)}")
    chunk = chunks[args.chunk]
    out_dir = args.out_dir or (DEFAULT_OUT_DIR / "eval" / args.chunk)

    config = p3run.load_config(args.config)
    base_fx = hz.load_fixtures()
    arm_c_fx = arm_c.arm_c_fixtures(base_fx)
    manifest = hz.load_manifest(args.data_dir)
    tasks = tasks_for_chunk(manifest, chunk)

    out_dir.mkdir(parents=True, exist_ok=True)
    hz.start_run(config, out_dir, receipt_set="phase3b_eval")

    template_fn = hz.build_hf_template_fn(config)
    generate_fn = hz.build_hf_generate_fn(config)
    records = run_tasks(tasks, base_fx, arm_c_fx, template_fn, generate_fn, config, args.data_dir)

    out_name = "records.jsonl" if chunk["kind"] == "agent" else "transcription_records.jsonl"
    hz.write_records(records, out_dir / out_name)
    print(f"[{args.chunk}] wrote {len(records)} {chunk['kind']} records to {out_dir / out_name}")


if __name__ == "__main__":
    main()
