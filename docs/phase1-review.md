# Phase 1 Review / Fix Pass

Date: 2026-08-18
Scope: fix pass only, per user request. No real OLMo evaluation started.
No git commits made.

---

## 1. PolyGuardPrompts label-count discrepancy — root cause, correction, and corrected numbers

### What was wrong

`docs/phase1-notes.md` (original) claimed, for the Korean subset:

- 945 unharmful + 754 harmful = 1,699 — but described this as covering all
  1,725 items, without accounting for the missing 26.
- Separately claimed "98.49% agreement (1,699/1,725)" between English and
  Korean `prompt_harm_label`, describing it as an *independent replication*
  of the source paper's reported 98.4% aggregate figure.

These two numbers look consistent with each other (both involve 1,699) but
are actually two different bugs that happened to produce numbers in the same
ballpark, not one bug. Recomputing from scratch below.

### Step 1 — exact `value_counts(dropna=False)` for both languages

```python
>>> df = pd.read_parquet('data/cache/polyguard_prompts.parquet')
>>> ko = df[df['language']=='Korean']
>>> en = df[df['language']=='English']
>>> ko['prompt_harm_label'].value_counts(dropna=False)
prompt_harm_label
unharmful    945
harmful      754
None          26
Name: count, dtype: int64

>>> en['prompt_harm_label'].value_counts(dropna=False)
prompt_harm_label
unharmful    945
harmful      754
None          26
Name: count, dtype: int64
```

945 + 754 + 26 = **1,725**, exactly. The original Phase 1 notes reported the
945/754 split correctly but never ran `dropna=False`, so the 26 `None` rows
were silently dropped from view and the text read as if 945+754 covered the
whole subset.

### Step 2 — what the 26 missing-label rows are

```python
>>> null_ko = ko[ko['prompt_harm_label'].isna()]
>>> sorted(null_ko['id'].tolist()) == sorted(en[en['prompt_harm_label'].isna()]['id'].tolist())
True   # identical 26 ids missing in both languages
>>> null_ko['prompt_label'].value_counts(dropna=False)
prompt_label
safe      20
unsafe     6
Name: count, dtype: int64
>>> null_ko['prompt_harm_agreement'].isna().all()
True   # all 26 also have a null prompt_harm_agreement
>>> null_ko['adversarial'].unique(), null_ko['subcategory'].unique()
(array([False]), array(['benign']))
```

These 26 items have a populated `prompt_label` (a coarser safe/unsafe field)
but no resolved `prompt_harm_label`, and `prompt_harm_agreement` is null for
all 26 — consistent with these being items where PolyGuard's own
annotator-agreement pipeline didn't converge on a harm label (the field this
project needs), even though a simpler label exists. All 26 are tagged
`adversarial=False, subcategory=benign` in both languages. This is a
property of the source dataset, not an artifact of this project's loader —
the same 26 ids are affected identically in both languages, which is itself
further evidence the `id` field is a genuine shared key across languages.

### Step 3 — EN–KO agreement, recomputed correctly

```python
>>> merged = ko[['id','prompt_harm_label']].merge(
...     en[['id','prompt_harm_label']], on='id', suffixes=('_ko','_en'))
>>> len(merged)
1725

>>> comparable = merged.dropna(subset=['prompt_harm_label_ko','prompt_harm_label_en'])
>>> len(comparable)
1699

>>> agree = comparable[comparable['prompt_harm_label_ko'] == comparable['prompt_harm_label_en']]
>>> len(agree)
1699

>>> len(comparable) - len(agree)   # disagreeing pairs
0
```

**Corrected result: 1,699/1,699 comparable pairs agree — 100%, zero
disagreements.** Not 98.49%.

### Step 4 — reproducing the original bug deliberately, to confirm the mechanism

```python
>>> naive = (merged['prompt_harm_label_ko'] == merged['prompt_harm_label_en'])
>>> naive.sum(), len(merged), naive.mean()
(1699, 1725, 0.9849275362318841)
```

This exactly reproduces the original 98.49% figure. The mechanism: pandas
(like standard IEEE float/NaN semantics) evaluates `NaN == NaN` as `False`.
The 26 rows where **both** sides are `None` were therefore counted as
*disagreements* by a naive `==` comparison over the full 1,725-row merge,
even though neither side actually contains conflicting information — both
sides simply have no label. `1699 / 1725 = 0.9849`, which happens to be
close to the paper's separately-reported 98.4% aggregate-across-17-languages
figure. That closeness is coincidental, not corroborating evidence — the
original write-up's claim of "independently reproducing the paper's figure"
does not hold up and has been removed.

### Step 5 — clearly distinguished final numbers

| Quantity | Value |
|---|---|
| Class balance (Korean, and identically English) | 945 unharmful / 754 harmful / 26 missing |
| Comparable EN–KO pairs (both sides labeled) | 1,699 |
| Agreeing pairs | 1,699 |
| Disagreeing pairs | 0 |
| Pairs with a missing label on at least one side | 26 (excluded from agreement calc — not counted as either agreeing or disagreeing) |
| Agreement rate (of comparable pairs) | **100%** (not 98.49%) |

### Step 6 — corrections applied

- `docs/phase1-notes.md` sections 1–2: rewritten with a correction notice,
  corrected numbers, and an explicit statement that the earlier "independent
  replication" claim was invalid.
- `configs/benchmarks/registry.yaml`, `polyguard_prompts.resolved_in_phase1`:
  replaced the single `korean_label_balance`/`provenance` fields with a full
  `korean_prompt_harm_label_value_counts` (including `missing: 26`) and a
  separate `en_ko_label_agreement` block with `comparable_pairs`,
  `agreeing_pairs`, `disagreeing_pairs`, `agreement_rate`, and
  `pairs_with_missing_label`, so the distinction this review draws can't
  collapse back into a single ambiguous number in the future.

### Practical implication for Phase 2A

None of this changes the Phase 2A plan (PolyGuardPrompts EN/KO + Belebele
EN/KO). If anything, the corrected finding is a *stronger* basis for using
PolyGuardPrompts than what was originally (wrongly) reported: on every item
where both languages have a resolved harm label, they agree with zero
exceptions in this data, rather than ~98.5%. The 26 unresolved-label items
should simply be excluded from any harm-label-based analysis (both prompt
and response harm scoring), consistent with how `expected_refusal()` in
`mpe/metrics/core.py` already returns `None` for any `expected_label` it
doesn't recognize (it will return `None` for these 26 once real generation
runs happen, correctly excluding them via `aggregate_rate`'s existing
None-filtering behavior — no code change was needed for this, only the
documentation was wrong).

---

## 2. MockEvaluator synthetic stage bias — fixed

Implemented **Option B**: default is neutral, opt-in is unmistakably named.

- `MockEvaluator(noise_rate=0.1)` (default construction) now applies the
  *same* noise distribution to every stage — verified by a new test,
  `test_effective_noise_identical_across_all_stages_by_default`.
- The bias table was renamed `TEST_ONLY_SYNTHETIC_STAGE_BIAS` (module-level
  constant, `mpe/evaluators/mock.py`) and is only applied when the evaluator
  is constructed with `use_test_only_synthetic_stage_bias=True`.
- Enabling it emits a `UserWarning` naming it a "FAKE stage-over-stage
  trend" and stating it must not be reported as a finding — so it cannot be
  turned on silently even by someone skimming a config or script.
- `scripts/run_experiment.py --evaluator mock` uses the neutral default; no
  CLI flag was added to expose the biased mode (deliberately — it's a
  debugging/demo knob, not something an experiment config should be able to
  quietly request).
- New test file `tests/unit/test_mock_evaluator.py` (7 tests) covers: neutral
  default, no warning by default, bias only applies when opted in, warning
  fires when opted in, and determinism of `generate()`.

Full suite: **72 passed** (was 65 — added 7 new MockEvaluator tests; no
existing test needed to change since none asserted on the old biased
behaviour).

---

## 3. Rule-based refusal parser — review

### 3.1 General failure mode

`classify_refusal()` in `mpe/parsing/output_parser.py` does regex
substring-matching for refusal phrases *anywhere* in the completion text.
The failure the smoke test surfaced (a compliant response that happened to
quote/echo a prompt whose own content was "I'm sorry, but I can't assist
with that") is one instance of a broader class: **the classifier cannot
distinguish "the model is asserting a refusal stance" from "refusal-shaped
language appears in the text for any other reason."** Concretely, this
includes:

- Echoing or quoting the prompt (the case observed).
- A hedge-then-comply pattern ("I understand this is sensitive, but here's
  how..." — contains no listed refusal phrase here, but the inverse failure
  is equally real: a genuine full refusal that also happens to include
  extra caveats a regex wasn't written to catch).
- Fictional/roleplay framing where a refusal phrase appears inside a story
  or dialogue the model is generating, not as its own stance.
- Partial compliance (refuses part of a multi-part request, complies with
  the rest) — the classifier forces a single binary label onto what may be
  a mixed response.
- Any refusal phrasing not in the fixed pattern list (the list is not
  exhaustive and was hand-written, not derived from real OLMo outputs).

This is a known, structural limitation of keyword/regex-based refusal
classification, not specific to this implementation — it's exactly the kind
of failure mode that motivated PolyGuard, WildGuard, and similar projects to
train dedicated classifiers instead of relying on pattern matching for
anything beyond a first pass.

### 3.2 Is it safe enough for Phase 2A as currently scoped?

**For PolyGuardPrompts: no, and it was never meant to be** — this was
already the design in `docs/phase0-findings.md` §6 (scoring hierarchy):
rule-based first, but PolyGuardPrompts specifically gets its own
benchmark-native classifier as primary scorer, precisely to avoid this class
of error. The rule-based classifier in `output_parser.py` remains useful for
Tier-1 checks (parseability, language-match) but was never the intended
authoritative scorer for PolyGuard's harm/refusal labels — the smoke-test
finding is a confirmation that this design decision was correct, not a new
problem.

**For Belebele**: not applicable — multiple-choice extraction
(`parse_mc_choice`) is a different, much more constrained task (finding a
digit 1–4 in a structural position) and is not exposed to this failure mode
in the same way. No change needed there.

**For XSTest**: moot for Phase 2A (XSTest is deferred to 2B per section 4
below). When it resumes, the rule-based classifier remains the closest thing
to a "native" scorer XSTest has (it ships none), but per the existing Phase
0 protocol it must be validated against a human-labeled subsample before
being trusted for reporting — this review is a concrete, observed reason
that validation step is load-bearing, not optional boilerplate.

### 3.3 Which scorer should be authoritative for PolyGuard

Confirmed (via direct HF API check, not assumed) that the PolyGuard project
ships actual classifier checkpoints, not just the prompt dataset:
`ToxicityPrompts/PolyGuard-Qwen`, `ToxicityPrompts/PolyGuard-Qwen-Smol`, and
`ToxicityPrompts/PolyGuard-Ministral`, all CC-BY-4.0. `PolyGuard-Qwen-Smol`
is a single-safetensors-file checkpoint (no sharding), making it the
lightest of the three and the natural default for this project's compute
budget — but this needs a Phase 2 sizing check (exact parameter count and a
latency estimate) before being locked in, since it's a real model that has
to be loaded and run, not a lookup table. Using one of these classifiers as
the authoritative PolyGuard scorer, with the current rule-based parser
retained only for Tier-1 validity checks, is the Phase 2A scoring plan.

### 3.4 What the rule-based parser should / should not be used for

**Should**: parseability checks (non-empty, non-degenerate output);
language-match detection; multiple-choice answer extraction (Belebele);
a cheap first-pass sanity filter before a benchmark-native or judge-based
scorer runs.

**Should not**: be the authoritative refusal/harm label for any benchmark
that has a validated native scorer available (PolyGuardPrompts); be trusted
for cross-lingual comparison claims without the validation step already
specified in Phase 0 for benchmarks that lack a native scorer (XSTest, when
it resumes).

**LLM judge**: not added, and not needed for Phase 2A as scoped (PolyGuard
has its native classifier; Belebele is exact-match). Consistent with "do not
add an LLM judge unless actually necessary" — it currently isn't.

---

## 4. XSTest scope decision + K-OverRefusal assessment

### 4.1 XSTest scope

No code or data changes made here — the 16-item draft in
`data/xstest_ko_draft/` was already explicitly marked
`draft_ai_translated_pending_native_review` and documented as a pipeline
fixture only (see `data/xstest_ko_draft/PROVENANCE.md`, written in Phase 1).
This review confirms that framing stays correct: **Phase 2A = PolyGuardPrompts
EN/KO + Belebele-KO control only. XSTest is Phase 2B**, gated on native-speaker
review of a properly-sized translated subsample, not on anything from this
review pass.

### 4.2 K-OverRefusal — could not be verified to exist

Searched directly (exact name, "K-OverRefusal" + huggingface/arxiv/github,
and general "Korean over-refusal benchmark 2026") and found no dataset,
paper, or repository by this name. One plausible near-miss surfaced —
**OKTest** — but that turned out to be "OverKill Test" (an English-language
over-refusal benchmark using the word "overkill," nothing to do with
Korean), so it's not a match. Two genuinely real, verified over-refusal
benchmarks did surface (OR-Bench, 80K English prompts; MORBENCH, a
multilingual over-refusal benchmark), neither of which is what was asked
about by name.

**I am not going to fabricate an assessment of a benchmark I can't confirm
exists** — that would be exactly the failure mode this project's whole
Phase 0/1 discipline has been built to avoid. Possibilities: it's extremely
recent and not yet indexed by general web search, it goes by a different
exact name I haven't guessed correctly, or it's from a venue/language my
search didn't surface (e.g. a Korean-language paper or a repo without much
English-language discussion linking to it yet).

**Request**: if you have a specific link, paper title, or HF/GitHub repo for
K-OverRefusal, share it and I'll do the same direct-verification pass on it
(item count, construct, license, scoring method, EN-KO parallelism) that
every other benchmark in this project has gone through, before making a
reject/could-have/replace recommendation.

**Conditional framework**, so this isn't a total non-answer: the trade-off
the user posed — XSTest's careful-but-effortful EN–KO parallelism vs. a
Korean-native over-refusal set's presumably-better cultural/linguistic
fidelity but weaker direct EN–KO comparability — maps onto this project's
core research question (does a post-training effect transfer *consistently*
across languages) as follows. A benchmark without genuine parallel EN/KO
items can still measure "does OLMo over-refuse in Korean" as a standalone
fact, but it cannot cleanly answer "is the over-refusal *change* from
post-training the same size in Korean as in English," because that
comparison needs the same underlying test item in both languages to control
for item-difficulty differences. So: if a verified K-OverRefusal turns out
to be Korean-only (no EN counterpart), it would most likely be a **could-have
extension** — a useful standalone Korean over-refusal measurement, and a
sanity check against XSTest-KO once that exists — rather than a **replacement**
for XSTest, precisely because replacing XSTest would give up the direct
cross-lingual delta this project is centrally about. It would only become a
credible **replacement** if it turns out to itself be a properly-parallel
EN–KO benchmark (in which case it might be a better choice than a
self-translated XSTest for exactly the provenance reasons this project cares
about) — which the current search couldn't confirm one way or the other.

---

## 5. Final verification

```
$ python3 -m pytest tests/ -q
72 passed in ~1s
```

(65 from Phase 1 + 7 new `MockEvaluator` tests from this review; 0 removed,
0 skipped.)

See the chat response for this turn for `git status` output and the full
summary of changes — this file focuses on the substance of the four review
items, not process bookkeeping.

### What's still blocking Phase 2A

Nothing new from this review. Standing items carried forward unchanged from
`docs/phase0-findings.md` / `docs/phase1-notes.md`:

- Cloud GPU compute decision + spend approval (needed before any real
  generation run, not before Phase 2A planning).
- HF token setup (recommended, not blocking, for bulk downloads).
- PolyGuard-Qwen-Smol sizing/latency check (new, small, identified in
  section 3.3 above) before it's wired in as the authoritative PolyGuard
  scorer.
