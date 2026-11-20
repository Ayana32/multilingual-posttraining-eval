#!/usr/bin/env python3
"""Phase 4 main-eval validation + H1/U1 scoring (docs/phase4-prereg.md sections 6, 7).

Closes the same gap the U2 scorer closes: before H1/U1 are judged on a run,
every planned manifest row (receipt x variant x channel) must be present exactly
once. This REUSES phase4/run_validation.validate_runs -- the identical function
phase4/score_u2.py uses -- then scores the collapsed, non-excluded records with
the FROZEN phase3/score.py (unchanged). If the planned denominator is not
complete, the criteria are reported inconclusive (never passed on a reduced
denominator).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "phase3"))
from run_validation import validate_runs
import score as sc  # frozen scorer, unchanged

CHANNELS = ("image", "text")


def planned_keys_for(manifest_rows):
    """Every manifest row runs on both channels (prereg section 6: 120 runs)."""
    return [(r["receipt_id"], r["variant"], ch) for r in manifest_rows for ch in CHANNELS]


def _key(r):
    return (r["receipt_id"], r["variant"], r["channel"])


def validate_and_score_main(records, manifest_rows, ground_truth):
    """Returns {validation, evaluable, cells, h1, u1}. `evaluable` is False when
    the planned denominator is incomplete; cells/H1/U1 are still reported but must
    not be treated as a pass."""
    v = validate_runs(records, planned_keys_for(manifest_rows), key_fn=_key)
    # score only the collapsed, non-excluded planned finals with the frozen scorer
    scored_records = [rec for k, rec in v["final"].items()
                      if k[0] in {r["receipt_id"] for r in manifest_rows}
                      and rec.get("outcome_status") != "harness_error"]
    scores = [sc.score_record(r, ground_truth) for r in scored_records]

    def pool(variant, measure, channels=CHANNELS):
        sel = [s for s in scores if s.variant == variant and s.channel in channels]
        return sum(1 for s in sel if getattr(s, measure)), len(sel)

    h1_k, h1_n = pool("P1_en", "M1")
    cells = {}
    for variant in ("clean", "P1_en", "P1_ko"):
        for ch in CHANNELS:
            sel = [s for s in scores if s.variant == variant and s.channel == ch]
            cells[f"{variant}|{ch}"] = {m: (sum(1 for s in sel if getattr(s, m)), len(sel))
                                        for m in ("M1", "M2", "M3", "TASK", "CO")}
    return {
        "validation": {k: v[k] for k in ("planned", "scored", "missing", "excluded",
                                         "missing_keys", "unexpected_keys", "duplicate_keys", "complete")},
        "evaluable": v["complete"],
        "h1_P1en_M1_pooled": {"k": h1_k, "n": h1_n},
        "u1_clean_TASK_pooled": {"k": pool("clean", "TASK")[0], "n": pool("clean", "TASK")[1]},
        "u1_P1en_TASK_pooled": {"k": pool("P1_en", "TASK")[0], "n": pool("P1_en", "TASK")[1]},
        "cells": cells,
    }


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Phase 4 main-eval validation + H1/U1")
    p.add_argument("records", type=Path)
    p.add_argument("--manifest", type=Path, default=REPO / "runs" / "phase4_manifest.jsonl")
    p.add_argument("--ground-truth", type=Path, default=REPO / "phase4" / "data" / "eval" / "receipts.jsonl")
    args = p.parse_args(argv)
    manifest_rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    records = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    gt = sc.load_ground_truth(args.ground_truth)
    out = validate_and_score_main(records, manifest_rows, gt)
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["evaluable"] else 2


if __name__ == "__main__":
    sys.exit(main())
