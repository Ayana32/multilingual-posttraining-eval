#!/usr/bin/env python3
"""Follow-up read-only analysis of the redacted Phase 2B records.
No files moved or modified.

By default reads data/phase2b/<run_id>/records_redacted.jsonl, including
the precomputed repetition metrics (dup_fraction, max_5gram_count,
is_collapse). Pass --raw-results-dir to recompute those metrics from the
real completion text instead (e.g. to validate the precomputed values,
or to re-run after a new generation); that is the only case in which
this script reads anything outside data/phase2b/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

RUN_IDS = {
    ("dpo", "en"): "phase2b_dpo_en_polyguard_17c33a34",
    ("dpo", "ko"): "phase2b_dpo_ko_polyguard_7b95ba0c",
    ("rlvr", "en"): "phase2b_rlvr_en_polyguard_151e7314",
    ("rlvr", "ko"): "phase2b_rlvr_ko_polyguard_324c0cc3",
}

STAGES = ["dpo", "rlvr"]
LANGS = ["en", "ko"]


def pct(n, d):
    return f"{n/d:.1%} ({n}/{d})" if d else "n/a (0/0)"


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def load_redacted(data_dir: Path) -> dict:
    data = {}
    for key, run_id in RUN_IDS.items():
        path = data_dir / run_id / "records_redacted.jsonl"
        with open(path, encoding="utf-8") as f:
            data[key] = [json.loads(line) for line in f if line.strip()]
    return data


def load_raw_repetition(raw_results_dir: Path) -> dict[tuple[str, str], dict]:
    """Recomputes repetition metrics from real completion text. Only used
    when --raw-results-dir is given. Keyed by (run's (stage, lang), item_id)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_redacted_records as redactor  # noqa: E402

    out = {}
    for key, run_id in RUN_IDS.items():
        path = raw_results_dir.expanduser() / run_id / "records.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                out[(key, r["item_id"])] = redactor.repetition_score(r["completion"])
    return out


def run(data_dir: Path, raw_results_dir: Path | None) -> None:
    data = load_redacted(data_dir)
    raw_repetition = load_raw_repetition(raw_results_dir) if raw_results_dir else None

    # -----------------------------------------------------------------------
    # 1. Refusal/compliance using expected_label as the denominator
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("1. Refusal / compliance rates using expected_label as denominator")
    print("=" * 78)
    print(f"{'stage':<6}{'lang':<5}{'refusal|expected_harmful':>28}  {'comply|expected_unharmful':>28}")
    for stage in STAGES:
        for lang in LANGS:
            records = data[(stage, lang)]
            harmful = [r for r in records if r["expected_label"] == "harmful"]
            unharmful = [r for r in records if r["expected_label"] == "unharmful"]
            n_refuse = sum(1 for r in harmful if r["scorer_response_refusal"] is True)
            n_comply = sum(1 for r in unharmful if r["scorer_response_refusal"] is False)
            print(f"{stage:<6}{lang:<5}{pct(n_refuse, len(harmful)):>28}  {pct(n_comply, len(unharmful)):>28}")

    # -----------------------------------------------------------------------
    # 2. Scorer agreement: scorer_prompt_harmful vs expected_label, per language
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("2. Scorer agreement: scorer_prompt_harmful vs expected_label")
    print("=" * 78)
    for lang in LANGS:
        for stage in STAGES:
            records = data[(stage, lang)]
            both_known = [r for r in records if r["expected_label"] is not None and r["scorer_prompt_harmful"] is not None]
            agree = sum(
                1 for r in both_known
                if (r["scorer_prompt_harmful"] is True) == (r["expected_label"] == "harmful")
            )
            tab = Counter((r["expected_label"], r["scorer_prompt_harmful"]) for r in both_known)
            print(f"-- stage={stage} lang={lang} -- agreement: {pct(agree, len(both_known))}")
            print(f"   {'expected_label':<16}{'scorer_prompt_harmful':<24}{'count':>6}")
            for (exp, sph), c in sorted(tab.items(), key=lambda kv: str(kv[0])):
                print(f"   {str(exp):<16}{str(sph):<24}{c:>6}")
        print()

    # -----------------------------------------------------------------------
    # 3. Harmful response rate among expected_label=harmful
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("3. Harmful response rate (scorer_response_harmful=true | expected_label=harmful)")
    print("=" * 78)
    print(f"{'stage':<6}{'lang':<5}{'harmful_response_rate':>26}")
    for stage in STAGES:
        for lang in LANGS:
            records = data[(stage, lang)]
            harmful = [r for r in records if r["expected_label"] == "harmful"]
            n_harmful_resp = sum(1 for r in harmful if r["scorer_response_harmful"] is True)
            print(f"{stage:<6}{lang:<5}{pct(n_harmful_resp, len(harmful)):>26}")

    # -----------------------------------------------------------------------
    # 4. Paired cross-lingual analysis (EN vs KO by parallel_item_id), harmful only
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("4. Paired EN-vs-KO cross-lingual analysis (expected_label=harmful only)")
    print("=" * 78)
    for stage in STAGES:
        en_records = {r["parallel_item_id"]: r for r in data[(stage, "en")] if r["expected_label"] == "harmful"}
        ko_records = {r["parallel_item_id"]: r for r in data[(stage, "ko")] if r["expected_label"] == "harmful"}
        shared_ids = sorted(set(en_records) & set(ko_records))

        a = b = c = d = excluded = 0
        for pid in shared_ids:
            ev, kv = en_records[pid]["scorer_response_refusal"], ko_records[pid]["scorer_response_refusal"]
            if ev is None or kv is None:
                excluded += 1
                continue
            if ev and kv:
                a += 1
            elif ev and not kv:
                b += 1
            elif not ev and kv:
                c += 1
            else:
                d += 1

        p = exact_mcnemar_p(b, c)
        print(f"-- stage={stage} -- shared harmful-prompt items: {len(shared_ids)} (excluded for None verdict: {excluded})")
        print(f"   {'':>18}{'KO refuse':>12}{'KO comply':>12}")
        print(f"   {'EN refuse':<18}{a:>12}{b:>12}")
        print(f"   {'EN comply':<18}{c:>12}{d:>12}")
        print(f"   discordant: EN-only-refuse(b)={b}  KO-only-refuse(c)={c}  McNemar exact p={p:.4g}")
        print()

    # -----------------------------------------------------------------------
    # 5. Truncation breakdown + repetition-collapse detection
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("5. Truncation breakdown (finish_reason=length) by expected_label x refusal_label")
    print("=" * 78)
    for stage in STAGES:
        for lang in LANGS:
            records = data[(stage, lang)]
            truncated = [r for r in records if r["finish_reason"] == "length"]
            tab = Counter((r["expected_label"], r["refusal_label"]) for r in truncated)
            print(f"-- stage={stage} lang={lang} -- truncated: {len(truncated)}/{len(records)}")
            if truncated:
                print(f"   {'expected_label':<16}{'refusal_label':<16}{'count':>6}")
                for (exp, rl), c in sorted(tab.items(), key=lambda kv: str(kv[0])):
                    print(f"   {str(exp):<16}{str(rl):<16}{c:>6}")
            print()

    print("=" * 78)
    print("5b. Repetition-collapse detection among truncated KO records")
    if raw_repetition is not None:
        print("    (recomputed from raw completion text via --raw-results-dir)")
    else:
        print("    (precomputed metrics from the redacted records)")
    print("=" * 78)
    for stage in STAGES:
        records = data[(stage, "ko")]
        truncated = [r for r in records if r["finish_reason"] == "length"]
        collapse_items = []
        normal_items = []
        for r in truncated:
            score = raw_repetition[((stage, "ko"), r["item_id"])] if raw_repetition is not None else {
                "dup_fraction": r["dup_fraction"], "max_5gram_count": r["max_5gram_count"], "is_collapse": r["is_collapse"]
            }
            if score["is_collapse"]:
                collapse_items.append((r["item_id"], score))
            else:
                normal_items.append((r["item_id"], score))

        print(f"-- stage={stage} lang=ko -- truncated: {len(truncated)}  "
              f"repetition_collapse: {len(collapse_items)}  normal_long_output: {len(normal_items)}")
        print(f"   example repetition_collapse item_ids: {[iid for iid, _ in collapse_items[:3]]}")
        for iid, s in collapse_items[:3]:
            print(f"     {iid}: dup_fraction={s['dup_fraction']:.2f} max_5gram_count={s['max_5gram_count']}")
        print(f"   example normal_long_output item_ids: {[iid for iid, _ in normal_items[:3]]}")
        for iid, s in normal_items[:3]:
            print(f"     {iid}: dup_fraction={s['dup_fraction']:.2f} max_5gram_count={s['max_5gram_count']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/phase2b"))
    parser.add_argument("--raw-results-dir", type=Path, default=None,
                         help="If given, recompute repetition metrics from real completion text instead of "
                              "using the precomputed fields in the redacted records.")
    args = parser.parse_args()
    run(args.data_dir, args.raw_results_dir)


if __name__ == "__main__":
    main()
