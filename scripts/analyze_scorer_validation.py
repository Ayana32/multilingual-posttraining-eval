#!/usr/bin/env python3
"""Analyze the completed EN and KO scorer-validation annotation sheets:
agreement/kappa/PABAK, population-reweighted human rate estimates with
two bootstrap methods (independent stratified, paired two-phase),
EN-KO comparison, exact McNemar, and paired-item cross-tabs.

Read-only on results/; writes only report/en_scorer_validation_completed.csv
(cleaned copy of the user's completed EN sheet, title row stripped) and
report/scorer_validation_summary.txt (this script's full output). Does
not modify either blind sheet. No commits.
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

REPORT_DIR = Path("~/multilingual-posttraining-eval/report").expanduser()

EN_RESULT_FILES = {
    "dpo": "~/results/phase2b_dpo_en_polyguard_17c33a34/records.jsonl",
    "rlvr": "~/results/phase2b_rlvr_en_polyguard_151e7314/records.jsonl",
}
KO_RESULT_FILES = {
    "dpo": "~/results/phase2b_dpo_ko_polyguard_7b95ba0c/records.jsonl",
    "rlvr": "~/results/phase2b_rlvr_ko_polyguard_324c0cc3/records.jsonl",
}

N_RESAMPLES = 20_000


# ---------------------------------------------------------------------------
# Loading / joining
# ---------------------------------------------------------------------------

def clean_en_completed_sheet() -> Path:
    src = Path("~/Downloads/en_scorer_validation_completed.csv").expanduser()
    dst = REPORT_DIR / "en_scorer_validation_completed.csv"
    text = src.read_text(encoding="utf-8-sig")
    first_line, rest = text.split("\n", 1)
    assert first_line.strip() == "en_scorer_validation", f"unexpected title row: {first_line!r}"
    dst.write_text(rest, encoding="utf-8-sig", newline="")
    rows = list(csv.DictReader(open(dst, encoding="utf-8-sig")))
    expected_cols = ["annotation_id", "prompt", "completion", "finish_reason",
                      "human_refusal", "human_harmful", "notes"]
    assert len(rows) == 30, f"expected 30 rows, got {len(rows)}"
    assert list(rows[0].keys()) == expected_cols, rows[0].keys()
    return dst


def load_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def yn(v: str) -> bool:
    return v.strip().lower() == "yes"


def tf(v: str) -> bool:
    return v.strip().lower() == "true"


def load_harmful_population(files: dict[str, str]) -> dict[str, list[dict]]:
    out = {}
    for stage, path_str in files.items():
        path = Path(path_str).expanduser()
        records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        out[stage] = [r for r in records if r["expected_label"] == "harmful"]
    return out


def bucket_of(r: dict) -> str:
    if r["scorer_response_refusal"] is True:
        return "refusal"
    if r["scorer_response_refusal"] is False and r["scorer_response_harmful"] is True:
        return "harmful_only"
    return "clean"


def join_sheet(completed_rows: list[dict], key_rows: list[dict], item_id_to_pid: dict[str, str] | None = None) -> list[dict]:
    key_by_id = {r["annotation_id"]: r for r in key_rows}
    assert {r["annotation_id"] for r in completed_rows} == set(key_by_id), "annotation_id mismatch"
    item_id_to_pid = item_id_to_pid or {}
    joined = []
    for c in completed_rows:
        k = key_by_id[c["annotation_id"]]
        pid = k.get("parallel_item_id") or item_id_to_pid.get(k["item_id"])
        joined.append({
            "annotation_id": c["annotation_id"],
            "item_id": k["item_id"],
            "parallel_item_id": pid,
            "stage": k["stage"],
            "bucket": k.get("bucket") or bucket_of({
                "scorer_response_refusal": tf(k["scorer_response_refusal"]),
                "scorer_response_harmful": tf(k["scorer_response_harmful"]),
            }),
            "human_refusal": yn(c["human_refusal"]),
            "human_harmful": yn(c["human_harmful"]),
            "scorer_refusal": tf(k["scorer_response_refusal"]),
            "scorer_harmful": tf(k["scorer_response_harmful"]),
            "notes": c["notes"],
        })
    return joined


# ---------------------------------------------------------------------------
# Agreement statistics: prevalence, observed/positive/negative agreement,
# Cohen's kappa, PABAK. Confusion matrix kept alongside.
# ---------------------------------------------------------------------------

def agreement_stats(pairs: list[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    tab = Counter(pairs)
    a, b, c, d = tab[(True, True)], tab[(True, False)], tab[(False, True)], tab[(False, False)]
    po = (a + d) / n
    prevalence_human = (a + b) / n
    prevalence_scorer = (a + c) / n
    pa = (2 * a) / (2 * a + b + c) if (2 * a + b + c) > 0 else float("nan")
    na = (2 * d) / (2 * d + b + c) if (2 * d + b + c) > 0 else float("nan")
    pe = prevalence_human * prevalence_scorer + (1 - prevalence_human) * (1 - prevalence_scorer)
    kappa = float("nan") if pe == 1.0 else (po - pe) / (1 - pe)
    pabak = 2 * po - 1
    return {
        "n": n, "confusion": tab,
        "prevalence_human": prevalence_human, "prevalence_scorer": prevalence_scorer,
        "observed_agreement": po, "positive_agreement": pa, "negative_agreement": na,
        "kappa": kappa, "pabak": pabak,
    }


def print_confusion(tab: Counter):
    print(f"    {'':>14}{'scorer=yes':>12}{'scorer=no':>12}")
    print(f"    {'human=yes':<14}{tab[(True, True)]:>12}{tab[(True, False)]:>12}")
    print(f"    {'human=no':<14}{tab[(False, True)]:>12}{tab[(False, False)]:>12}")


def print_agreement_block(label: str, joined: list[dict]):
    print(f"-- {label} --")
    for metric, hk, sk in (("refusal", "human_refusal", "scorer_refusal"), ("harmful", "human_harmful", "scorer_harmful")):
        pairs = [(r[hk], r[sk]) for r in joined]
        s = agreement_stats(pairs)
        print(f"  {metric}: n={s['n']}  prevalence(human)={s['prevalence_human']:.1%}  "
              f"prevalence(scorer)={s['prevalence_scorer']:.1%}")
        print(f"    observed_agreement={s['observed_agreement']:.1%}  "
              f"positive_agreement={s['positive_agreement']:.1%}  "
              f"negative_agreement={s['negative_agreement']:.1%}")
        print(f"    kappa={s['kappa']:.3f}  PABAK={s['pabak']:.3f}")
        print_confusion(s["confusion"])
    print()


# ---------------------------------------------------------------------------
# Population-reweighted estimate + independent stratified bootstrap
# ---------------------------------------------------------------------------

def bucket_samples(joined: list[dict], metric_key: str) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for r in joined:
        out.setdefault(r["bucket"], []).append(r[metric_key])
    return out


def reweighted_estimate(bucket_vals: dict[str, list[bool]], population: dict[str, int]) -> float:
    total_pop = sum(population.values())
    est = 0.0
    for bucket, pop_n in population.items():
        vals = bucket_vals.get(bucket, [])
        if not vals:
            continue
        rate = sum(vals) / len(vals)
        est += (pop_n / total_pop) * rate
    return est


def bootstrap_reweighted(bucket_vals: dict[str, list[bool]], population: dict[str, int],
                           n_resamples: int, rng: random.Random) -> list[float]:
    draws = []
    bucket_names = list(population.keys())
    for _ in range(n_resamples):
        resampled_vals = {}
        for b in bucket_names:
            vals = bucket_vals.get(b, [])
            resampled_vals[b] = [rng.choice(vals) for _ in vals] if vals else []
        draws.append(reweighted_estimate(resampled_vals, population))
    return draws


# ---------------------------------------------------------------------------
# Paired two-phase bootstrap for the EN-KO human refusal gap.
# Outer: resample the 186 matched (stage, parallel_item_id) pairs with
# replacement, keeping each pair intact -- this re-derives new per-language
# bucket compositions each draw (since EN and KO scorer verdicts on the
# same item can put it in different buckets).
# Inner: resample each language's own validation sample within bucket, as
# in the independent method, to get a bucket human rate.
# ---------------------------------------------------------------------------

def paired_two_phase_bootstrap(matched_keys, en_by_key_full, ko_by_key_full,
                                 en_bucket_vals, ko_bucket_vals,
                                 n_resamples, seed):
    """Returns (en_human_draws, ko_human_draws, en_scorer_draws, ko_scorer_draws).
    The scorer draws need no inner resampling -- scorer_response_refusal is
    exactly True for the 'refusal' bucket and exactly False for the other two
    by construction (bucket_of), so the scorer refusal rate on a resampled
    population is just that population's 'refusal' bucket share."""
    rng = random.Random(seed)
    n = len(matched_keys)
    en_human_draws, ko_human_draws = [], []
    en_scorer_draws, ko_scorer_draws = [], []
    for _ in range(n_resamples):
        resampled_keys = [matched_keys[rng.randrange(n)] for _ in range(n)]
        en_counts = Counter(bucket_of(en_by_key_full[k]) for k in resampled_keys)
        ko_counts = Counter(bucket_of(ko_by_key_full[k]) for k in resampled_keys)

        en_rate = 0.0
        for b, cnt in en_counts.items():
            vals = en_bucket_vals.get(b, [])
            if not vals:
                continue
            inner = [rng.choice(vals) for _ in vals]
            en_rate += (cnt / n) * (sum(inner) / len(inner))

        ko_rate = 0.0
        for b, cnt in ko_counts.items():
            vals = ko_bucket_vals.get(b, [])
            if not vals:
                continue
            inner = [rng.choice(vals) for _ in vals]
            ko_rate += (cnt / n) * (sum(inner) / len(inner))

        en_human_draws.append(en_rate)
        ko_human_draws.append(ko_rate)
        en_scorer_draws.append(en_counts.get("refusal", 0) / n)
        ko_scorer_draws.append(ko_counts.get("refusal", 0) / n)
    return en_human_draws, ko_human_draws, en_scorer_draws, ko_scorer_draws


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def ci(draws: list[float]):
    return percentile(draws, 2.5), percentile(draws, 97.5)


def true_scorer_rate(population_records: list[dict], key: str) -> float:
    return sum(1 for r in population_records if r[key] is True) / len(population_records)


def jeffreys_robustness(bucket_vals: dict[str, list[bool]], population: dict[str, int],
                         n_draws: int, seed: int) -> list[float]:
    """Per bucket, draws the human refusal rate from Beta(k+0.5, n-k+0.5)
    (Jeffreys prior Beta(0.5, 0.5) updated with the validation sample's
    k successes out of n), combined with the fixed, exact population
    bucket counts (no resampling of population composition). Unlike the
    within-bucket resampling bootstrap, this assigns nonzero uncertainty
    to a bucket's rate even when every sampled label in it is identical."""
    rng = random.Random(seed)
    total_pop = sum(population.values())
    draws = []
    for _ in range(n_draws):
        est = 0.0
        for bucket, pop_n in population.items():
            vals = bucket_vals.get(bucket, [])
            n = len(vals)
            if n == 0:
                continue
            k = sum(vals)
            rate = rng.betavariate(k + 0.5, n - k + 0.5)
            est += (pop_n / total_pop) * rate
        draws.append(est)
    return draws


def median(sorted_vals: list[float]) -> float:
    return percentile(sorted_vals, 50)


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _run():
    clean_en_completed_sheet()

    en_completed = load_csv(REPORT_DIR / "en_scorer_validation_completed.csv")
    en_key = load_csv(REPORT_DIR / "en_scorer_validation_key.csv")
    en_population_rows = load_csv(REPORT_DIR / "en_scorer_validation_population.csv")
    en_population = {"dpo": {}, "rlvr": {}}
    for row in en_population_rows:
        en_population[row["stage"]][row["bucket"]] = int(row["population_count"])
    en_population_pooled = {
        b: en_population["dpo"][b] + en_population["rlvr"][b] for b in ("refusal", "harmful_only", "clean")
    }

    ko_completed = load_csv(REPORT_DIR / "ko_scorer_validation_completed.csv")
    ko_key = load_csv(REPORT_DIR / "ko_scorer_validation_key.csv")
    ko_population_pooled = {"refusal": 21, "harmful_only": 31, "clean": 134}

    ko_full = load_harmful_population(KO_RESULT_FILES)
    en_full = load_harmful_population(EN_RESULT_FILES)
    en_full_pooled = [r for recs in en_full.values() for r in recs]
    ko_full_pooled = [r for recs in ko_full.values() for r in recs]
    ko_item_to_pid = {r["item_id"]: r["parallel_item_id"] for r in ko_full_pooled}

    en_joined = join_sheet(en_completed, en_key)
    ko_joined = join_sheet(ko_completed, ko_key, item_id_to_pid=ko_item_to_pid)

    # verify the given KO population counts against the real underlying data
    ko_true_pooled = Counter()
    for stage_records in ko_full.values():
        for r in stage_records:
            ko_true_pooled[bucket_of(r)] += 1
    assert dict(ko_true_pooled) == ko_population_pooled, (
        f"KO population mismatch: given={ko_population_pooled} computed={dict(ko_true_pooled)}"
    )

    # matched (stage, parallel_item_id) population for the paired bootstrap
    en_by_key_full = {(stage, r["parallel_item_id"]): r for stage, recs in en_full.items() for r in recs}
    ko_by_key_full = {(stage, r["parallel_item_id"]): r for stage, recs in ko_full.items() for r in recs}
    matched_keys = sorted(set(en_by_key_full) & set(ko_by_key_full))
    print(f"Matched (stage, parallel_item_id) population pairs: {len(matched_keys)}")
    en_marginal_check = Counter(bucket_of(en_by_key_full[k]) for k in matched_keys)
    ko_marginal_check = Counter(bucket_of(ko_by_key_full[k]) for k in matched_keys)
    assert dict(en_marginal_check) == en_population_pooled, (en_marginal_check, en_population_pooled)
    assert dict(ko_marginal_check) == ko_population_pooled, (ko_marginal_check, ko_population_pooled)
    print("Matched-population per-language bucket marginals match the independent population counts (verified).\n")

    print("=" * 78)
    print("1. Agreement statistics (prevalence, agreement, kappa, PABAK) + confusion tables")
    print("=" * 78)
    print_agreement_block("EN (n=30)", en_joined)
    print_agreement_block("KO (n=40)", ko_joined)

    en_refusal_vals = bucket_samples(en_joined, "human_refusal")
    en_harmful_vals = bucket_samples(en_joined, "human_harmful")
    ko_refusal_vals = bucket_samples(ko_joined, "human_refusal")
    ko_harmful_vals = bucket_samples(ko_joined, "human_harmful")

    en_refusal_point = reweighted_estimate(en_refusal_vals, en_population_pooled)
    en_harmful_point = reweighted_estimate(en_harmful_vals, en_population_pooled)
    ko_refusal_point = reweighted_estimate(ko_refusal_vals, ko_population_pooled)
    ko_harmful_point = reweighted_estimate(ko_harmful_vals, ko_population_pooled)

    en_refusal_draws = sorted(bootstrap_reweighted(en_refusal_vals, en_population_pooled, N_RESAMPLES, random.Random(42)))
    en_harmful_draws = sorted(bootstrap_reweighted(en_harmful_vals, en_population_pooled, N_RESAMPLES, random.Random(43)))
    ko_refusal_draws = sorted(bootstrap_reweighted(ko_refusal_vals, ko_population_pooled, N_RESAMPLES, random.Random(44)))
    ko_harmful_draws = sorted(bootstrap_reweighted(ko_harmful_vals, ko_population_pooled, N_RESAMPLES, random.Random(45)))

    en_scorer_refusal = true_scorer_rate(en_full_pooled, "scorer_response_refusal")
    en_scorer_harmful = true_scorer_rate(en_full_pooled, "scorer_response_harmful")
    ko_scorer_refusal = true_scorer_rate(ko_full_pooled, "scorer_response_refusal")
    ko_scorer_harmful = true_scorer_rate(ko_full_pooled, "scorer_response_harmful")

    print("=" * 78)
    print("2. EN population-reweighted human estimates vs scorer (n=186 harmful-label pop.)")
    print("=" * 78)
    lo, hi = ci(en_refusal_draws)
    print(f"  human refusal rate  : {en_refusal_point:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  scorer refusal rate : {en_scorer_refusal:.1%}  (true population value)")
    lo, hi = ci(en_harmful_draws)
    print(f"  human harmful rate  : {en_harmful_point:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  scorer harmful rate : {en_scorer_harmful:.1%}  (true population value)")
    print()

    print("=" * 78)
    print("3. KO population-reweighted human estimates vs scorer (n=186 harmful-label pop.)")
    print("=" * 78)
    print(f"  KO population given: {ko_population_pooled}  (verified against raw records: match)")
    lo, hi = ci(ko_refusal_draws)
    print(f"  human refusal rate  : {ko_refusal_point:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  scorer refusal rate : {ko_scorer_refusal:.1%}  (true population value)")
    lo, hi = ci(ko_harmful_draws)
    print(f"  human harmful rate  : {ko_harmful_point:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  scorer harmful rate : {ko_scorer_harmful:.1%}  (true population value)")
    print()

    # -- independent stratified bootstrap for the human-based refusal gap --
    indep_rng = random.Random(46)
    en_indep_draws, ko_indep_draws = [], []
    for _ in range(N_RESAMPLES):
        en_resampled = {b: [indep_rng.choice(v) for _ in v] if v else [] for b, v in en_refusal_vals.items()}
        ko_resampled = {b: [indep_rng.choice(v) for _ in v] if v else [] for b, v in ko_refusal_vals.items()}
        en_indep_draws.append(reweighted_estimate(en_resampled, en_population_pooled))
        ko_indep_draws.append(reweighted_estimate(ko_resampled, ko_population_pooled))
    gap_indep_draws = sorted(e - k for e, k in zip(en_indep_draws, ko_indep_draws))
    gap_indep_lo, gap_indep_hi = ci(gap_indep_draws)

    # -- paired two-phase bootstrap for the human-based refusal gap --
    en_paired_draws, ko_paired_draws, en_scorer_paired_draws, ko_scorer_paired_draws = paired_two_phase_bootstrap(
        matched_keys, en_by_key_full, ko_by_key_full, en_refusal_vals, ko_refusal_vals, N_RESAMPLES, seed=42
    )
    gap_paired_draws_unsorted = [e - k for e, k in zip(en_paired_draws, ko_paired_draws)]
    gap_paired_draws = sorted(gap_paired_draws_unsorted)
    gap_paired_lo, gap_paired_hi = ci(gap_paired_draws)

    gap_point = en_refusal_point - ko_refusal_point
    scorer_gap_point = en_scorer_refusal - ko_scorer_refusal

    # -- D = human gap - scorer gap, under both views --
    # Paired two-phase view: both the human gap and the scorer gap vary per
    # draw (bucket composition itself is resampled), so D is computed per
    # draw from the same outer resample before taking percentiles.
    scorer_gap_paired_draws_unsorted = [e - k for e, k in zip(en_scorer_paired_draws, ko_scorer_paired_draws)]
    D_paired_draws = sorted(h - s for h, s in zip(gap_paired_draws_unsorted, scorer_gap_paired_draws_unsorted))
    D_point = gap_point - scorer_gap_point
    D_paired_lo, D_paired_hi = ci(D_paired_draws)

    # Fixed-population view: bucket counts are fixed at their true values,
    # so the scorer gap is exact (not resampled) -- D's distribution is just
    # the independent-stratified human-gap distribution shifted by a
    # constant.
    D_indep_draws = sorted(g - scorer_gap_point for g in gap_indep_draws)
    D_indep_lo, D_indep_hi = ci(D_indep_draws)

    print("=" * 78)
    print("4. Summary table + human-based EN-KO refusal gap (two bootstrap methods) + scorer-based gap")
    print("=" * 78)
    print(f"{'lang':<6}{'scorer_refusal':>15}{'human_refusal':>26}{'scorer_harmful':>16}{'human_harmful':>26}")
    lo_r, hi_r = ci(en_refusal_draws)
    lo_h, hi_h = ci(en_harmful_draws)
    print(f"{'EN':<6}{en_scorer_refusal:>14.1%} {en_refusal_point:>9.1%} [{lo_r:.1%},{hi_r:.1%}]"
          f"{en_scorer_harmful:>15.1%} {en_harmful_point:>9.1%} [{lo_h:.1%},{hi_h:.1%}]")
    lo_r, hi_r = ci(ko_refusal_draws)
    lo_h, hi_h = ci(ko_harmful_draws)
    print(f"{'KO':<6}{ko_scorer_refusal:>14.1%} {ko_refusal_point:>9.1%} [{lo_r:.1%},{hi_r:.1%}]"
          f"{ko_scorer_harmful:>15.1%} {ko_harmful_point:>9.1%} [{lo_h:.1%},{hi_h:.1%}]")
    print()
    print(f"  human-based EN-KO refusal gap (point estimate): {gap_point:+.1%}")
    print(f"    independent stratified bootstrap (fixed-population view) : 95% CI [{gap_indep_lo:+.1%}, {gap_indep_hi:+.1%}]")
    print(f"    paired two-phase bootstrap (generalization view)         : 95% CI [{gap_paired_lo:+.1%}, {gap_paired_hi:+.1%}]")
    print(f"  scorer-based EN-KO refusal gap (exact, n=186 pop. each): {scorer_gap_point:+.1%}")
    print()
    print(f"  D = human gap - scorer gap, point estimate: {D_point:+.1%}")
    print(f"    fixed-population view (scorer gap exact)      : 95% CI [{D_indep_lo:+.1%}, {D_indep_hi:+.1%}]")
    print(f"    generalization view (scorer gap also resampled): 95% CI [{D_paired_lo:+.1%}, {D_paired_hi:+.1%}]")
    print()

    print("=" * 78)
    print("4b. Refusal error-direction table (from the refusal confusion matrices)")
    print("=" * 78)
    for label, joined in (("EN", en_joined), ("KO", ko_joined)):
        pairs = [(r["human_refusal"], r["scorer_refusal"]) for r in joined]
        s = agreement_stats(pairs)
        a, b, c, d = (s["confusion"][(True, True)], s["confusion"][(True, False)],
                      s["confusion"][(False, True)], s["confusion"][(False, False)])
        human_yes_total = a + b
        scorer_yes_total = a + c
        missed_rate = b / human_yes_total if human_yes_total else float("nan")
        unconfirmed_rate = c / scorer_yes_total if scorer_yes_total else float("nan")
        print(f"  {label}: human refusal missed by scorer       : {b}/{human_yes_total} ({missed_rate:.1%})")
        print(f"  {label}: scorer refusal not confirmed by human : {c}/{scorer_yes_total} ({unconfirmed_rate:.1%})")
    print()

    print("=" * 78)
    print("5. Every EN disagreement, and 'disclaimer-then-comply' counts by scorer label")
    print("=" * 78)
    en_disagreements = [r for r in en_joined if r["human_refusal"] != r["scorer_refusal"] or r["human_harmful"] != r["scorer_harmful"]]
    print(f"Total EN disagreements: {len(en_disagreements)}/{len(en_joined)}\n")
    for r in en_disagreements:
        notes_snip = r["notes"][:150].replace("\n", " ")
        print(f"  {r['annotation_id']}  item_id={r['item_id']}  stage={r['stage']}  bucket={r['bucket']}")
        print(f"    human: refusal={r['human_refusal']} harmful={r['human_harmful']}   "
              f"scorer: refusal={r['scorer_refusal']} harmful={r['scorer_harmful']}")
        print(f"    notes: {notes_snip}")
        print()

    dtc_rows = [r for r in en_joined if "disclaimer-then-comply" in r["notes"].lower()]
    print(f"'disclaimer-then-comply' mentions: {len(dtc_rows)}")
    by_scorer_refusal = Counter(r["scorer_refusal"] for r in dtc_rows)
    by_scorer_harmful = Counter(r["scorer_harmful"] for r in dtc_rows)
    print(f"  by scorer_refusal: {dict(by_scorer_refusal)}")
    print(f"  by scorer_harmful: {dict(by_scorer_harmful)}")
    print()

    print("=" * 78)
    print("5b. Every KO disagreement, and notes flagged misunderstanding/unclear")
    print("=" * 78)
    ko_disagreements = [r for r in ko_joined if r["human_refusal"] != r["scorer_refusal"] or r["human_harmful"] != r["scorer_harmful"]]
    print(f"Total KO disagreements: {len(ko_disagreements)}/{len(ko_joined)}\n")
    for r in ko_disagreements:
        notes_snip = r["notes"][:150].replace("\n", " ")
        print(f"  {r['annotation_id']}  item_id={r['item_id']}  stage={r['stage']}  bucket={r['bucket']}")
        print(f"    human: refusal={r['human_refusal']} harmful={r['human_harmful']}   "
              f"scorer: refusal={r['scorer_refusal']} harmful={r['scorer_harmful']}")
        print(f"    notes: {notes_snip}")
        print()

    ko_flagged = [r for r in ko_joined if r["notes"].strip().lower().startswith(("misunderstanding", "unclear"))]
    print(f"KO rows flagged misunderstanding/unclear: {len(ko_flagged)}")
    for r in ko_flagged:
        print(f"  {r['annotation_id']}  item_id={r['item_id']}  stage={r['stage']}  "
              f"human_harmful={r['human_harmful']}  scorer_harmful={r['scorer_harmful']}  "
              f"human_refusal={r['human_refusal']}  scorer_refusal={r['scorer_refusal']}")
    print()

    print("=" * 78)
    print("6. Paired items (same stage + parallel_item_id in both EN and KO validation samples)")
    print("=" * 78)
    en_by_key = {(r["stage"], r["parallel_item_id"]): r for r in en_joined}
    ko_by_key = {(r["stage"], r["parallel_item_id"]): r for r in ko_joined}
    shared_keys = sorted(set(en_by_key) & set(ko_by_key))
    print(f"Shared (stage, parallel_item_id) pairs: {len(shared_keys)}\n")
    tab = Counter((en_by_key[k]["human_refusal"], ko_by_key[k]["human_refusal"]) for k in shared_keys)
    print(f"  {'':>18}{'KO refuse':>12}{'KO comply':>12}")
    print(f"  {'EN refuse':<18}{tab[(True, True)]:>12}{tab[(True, False)]:>12}")
    print(f"  {'EN comply':<18}{tab[(False, True)]:>12}{tab[(False, False)]:>12}")
    b, c = tab[(True, False)], tab[(False, True)]
    p_mcnemar = exact_mcnemar_p(b, c)
    print(f"\n  discordant: EN-only-refuse(b)={b}  KO-only-refuse(c)={c}  exact McNemar p={p_mcnemar:.4g}")
    print()
    print("  pairs detail:")
    for k in shared_keys:
        stage, pid = k
        print(f"    stage={stage} parallel_item_id={pid}  EN_human_refusal={en_by_key[k]['human_refusal']}  "
              f"KO_human_refusal={ko_by_key[k]['human_refusal']}")

    print()
    print("=" * 78)
    print("7. Sensitivity analysis: strict refusal definition")
    print("=" * 78)
    print("Only explicit refusal statements count as refusal under this reading. E02, E15,")
    print("E26, E27 (the redirect-without-explicit-refusal items from sections 7 and 11) are")
    print("relabelled human_refusal=False; KO labels are unchanged.")
    print()

    strict_flip_ids = {"E02", "E15", "E26", "E27"}
    en_joined_strict = [
        {**r, "human_refusal": False} if r["annotation_id"] in strict_flip_ids else r
        for r in en_joined
    ]
    flipped = [r for r in en_joined_strict if r["annotation_id"] in strict_flip_ids]
    assert len(flipped) == 4 and all(r["human_refusal"] is False for r in flipped), flipped

    en_refusal_vals_strict = bucket_samples(en_joined_strict, "human_refusal")
    en_refusal_point_strict = reweighted_estimate(en_refusal_vals_strict, en_population_pooled)
    en_refusal_draws_strict = sorted(bootstrap_reweighted(en_refusal_vals_strict, en_population_pooled, N_RESAMPLES, random.Random(42)))
    lo_strict, hi_strict = ci(en_refusal_draws_strict)

    print(f"  human EN refusal rate")
    print(f"    original : {en_refusal_point:.1%}  95% CI [{ci(en_refusal_draws)[0]:.1%}, {ci(en_refusal_draws)[1]:.1%}]")
    print(f"    strict   : {en_refusal_point_strict:.1%}  95% CI [{lo_strict:.1%}, {hi_strict:.1%}]")
    print()

    # independent stratified, strict EN vals / unchanged KO vals -- same
    # seed as the original independent-stratified run (46): since the
    # per-bucket EN sample sizes are unchanged (only 4 labels flipped, not
    # added/removed), the rng stream position is identical at every step,
    # so ko_indep_draws_strict is byte-identical to ko_indep_draws and only
    # the EN side differs. That isolates the sensitivity to exactly the
    # four relabelled items.
    indep_rng_strict = random.Random(46)
    en_indep_draws_strict, ko_indep_draws_strict = [], []
    for _ in range(N_RESAMPLES):
        en_resampled = {b: [indep_rng_strict.choice(v) for _ in v] if v else [] for b, v in en_refusal_vals_strict.items()}
        ko_resampled = {b: [indep_rng_strict.choice(v) for _ in v] if v else [] for b, v in ko_refusal_vals.items()}
        en_indep_draws_strict.append(reweighted_estimate(en_resampled, en_population_pooled))
        ko_indep_draws_strict.append(reweighted_estimate(ko_resampled, ko_population_pooled))
    gap_indep_draws_strict = sorted(e - k for e, k in zip(en_indep_draws_strict, ko_indep_draws_strict))
    gap_indep_lo_strict, gap_indep_hi_strict = ci(gap_indep_draws_strict)
    gap_point_strict = en_refusal_point_strict - ko_refusal_point

    # paired two-phase, strict EN vals / unchanged KO vals, same seed (42)
    en_paired_draws_strict, ko_paired_draws_strict, en_scorer_paired_draws_strict, ko_scorer_paired_draws_strict = paired_two_phase_bootstrap(
        matched_keys, en_by_key_full, ko_by_key_full, en_refusal_vals_strict, ko_refusal_vals, N_RESAMPLES, seed=42
    )
    assert en_scorer_paired_draws_strict == en_scorer_paired_draws and ko_scorer_paired_draws_strict == ko_scorer_paired_draws, (
        "scorer draws should be unaffected by the strict relabelling -- only human draws should differ"
    )
    gap_paired_draws_unsorted_strict = [e - k for e, k in zip(en_paired_draws_strict, ko_paired_draws_strict)]
    gap_paired_draws_strict = sorted(gap_paired_draws_unsorted_strict)
    gap_paired_lo_strict, gap_paired_hi_strict = ci(gap_paired_draws_strict)

    print(f"  human-based EN-KO refusal gap")
    print(f"    original, fixed-population      : {gap_point:+.1%}  95% CI [{gap_indep_lo:+.1%}, {gap_indep_hi:+.1%}]")
    print(f"    strict,   fixed-population       : {gap_point_strict:+.1%}  95% CI [{gap_indep_lo_strict:+.1%}, {gap_indep_hi_strict:+.1%}]")
    print(f"    original, generalization         : {gap_point:+.1%}  95% CI [{gap_paired_lo:+.1%}, {gap_paired_hi:+.1%}]")
    print(f"    strict,   generalization         : {gap_point_strict:+.1%}  95% CI [{gap_paired_lo_strict:+.1%}, {gap_paired_hi_strict:+.1%}]")
    print()

    # D under the strict reading, both views
    D_point_strict = gap_point_strict - scorer_gap_point
    D_indep_draws_strict = sorted(g - scorer_gap_point for g in gap_indep_draws_strict)
    D_indep_lo_strict, D_indep_hi_strict = ci(D_indep_draws_strict)
    scorer_gap_paired_draws_unsorted = [e - k for e, k in zip(en_scorer_paired_draws, ko_scorer_paired_draws)]
    D_paired_draws_strict = sorted(h - s for h, s in zip(gap_paired_draws_unsorted_strict, scorer_gap_paired_draws_unsorted))
    D_paired_lo_strict, D_paired_hi_strict = ci(D_paired_draws_strict)

    print(f"  D = human gap - scorer gap")
    print(f"    original, fixed-population : {D_point:+.1%}  95% CI [{D_indep_lo:+.1%}, {D_indep_hi:+.1%}]  excludes 0: {D_indep_lo > 0}")
    print(f"    strict,   fixed-population  : {D_point_strict:+.1%}  95% CI [{D_indep_lo_strict:+.1%}, {D_indep_hi_strict:+.1%}]  excludes 0: {D_indep_lo_strict > 0}")
    print(f"    original, generalization   : {D_point:+.1%}  95% CI [{D_paired_lo:+.1%}, {D_paired_hi:+.1%}]  excludes 0: {D_paired_lo > 0}")
    print(f"    strict,   generalization    : {D_point_strict:+.1%}  95% CI [{D_paired_lo_strict:+.1%}, {D_paired_hi_strict:+.1%}]  excludes 0: {D_paired_lo_strict > 0}")

    print()
    print("=" * 78)
    print("8. Jeffreys-prior robustness check (fixed population bucket counts, no resampling")
    print("   of population composition; Beta(k+0.5, n-k+0.5) per bucket; 40000 draws, seed 42)")
    print("=" * 78)

    N_JEFFREYS = 40_000
    # Unsorted, per-iteration order preserved -- required for correct
    # pairing when computing the gap and D below. Sorting before zipping
    # would pair the i-th smallest EN draw with the i-th smallest KO draw,
    # inducing spurious positive correlation and an artificially narrow
    # gap/D distribution.
    en_jeff_inclusive_raw = jeffreys_robustness(en_refusal_vals, en_population_pooled, N_JEFFREYS, seed=42)
    en_jeff_strict_raw = jeffreys_robustness(en_refusal_vals_strict, en_population_pooled, N_JEFFREYS, seed=42)
    ko_jeff_raw = jeffreys_robustness(ko_refusal_vals, ko_population_pooled, N_JEFFREYS, seed=43)

    en_jeff_inclusive = sorted(en_jeff_inclusive_raw)
    en_jeff_strict = sorted(en_jeff_strict_raw)
    ko_jeff = sorted(ko_jeff_raw)

    def report(label, draws):
        lo, hi = ci(draws)
        print(f"  {label}: median={median(draws):.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

    print("-- EN human refusal rate --")
    report("inclusive", en_jeff_inclusive)
    report("strict   ", en_jeff_strict)
    print("-- KO human refusal rate (same under both readings) --")
    report("KO       ", ko_jeff)

    gap_jeff_inclusive = sorted(e - k for e, k in zip(en_jeff_inclusive_raw, ko_jeff_raw))
    gap_jeff_strict = sorted(e - k for e, k in zip(en_jeff_strict_raw, ko_jeff_raw))
    print("-- human-based EN-KO refusal gap --")
    report("inclusive", gap_jeff_inclusive)
    report("strict   ", gap_jeff_strict)

    D_jeff_inclusive = sorted(g - scorer_gap_point for g in gap_jeff_inclusive)
    D_jeff_strict = sorted(g - scorer_gap_point for g in gap_jeff_strict)
    print("-- D = human gap - scorer gap (scorer gap fixed at exact population value) --")
    lo, hi = ci(D_jeff_inclusive)
    print(f"  inclusive: median={median(D_jeff_inclusive):+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  excludes 0: {lo > 0}")
    lo, hi = ci(D_jeff_strict)
    print(f"  strict   : median={median(D_jeff_strict):+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  excludes 0: {lo > 0}")
    print()

    print("-- Buckets with all-identical sampled labels (zero bootstrap variance), by language/reading --")
    for label, vals_dict in (
        ("EN inclusive", en_refusal_vals),
        ("EN strict", en_refusal_vals_strict),
        ("KO", ko_refusal_vals),
    ):
        degenerate = []
        for bucket, vals in vals_dict.items():
            if vals and len(set(vals)) == 1:
                k = sum(vals)
                degenerate.append(f"{bucket} ({k}/{len(vals)})")
        print(f"  {label}: {', '.join(degenerate) if degenerate else 'none'}")


def main():
    out_path = REPORT_DIR / "scorer_validation_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(Tee(sys.stdout, f)):
            _run()
    print(f"\n(Full output also written to {out_path})")


if __name__ == "__main__":
    main()
