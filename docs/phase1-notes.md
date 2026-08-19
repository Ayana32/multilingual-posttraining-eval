# Phase 1 Notes — Open Item Resolution + Evaluation Core

Date: 2026-08-18
Status: Phase 1 complete, pending user review. Stops before Phase 2 (no real
7B evaluation run has happened yet — everything below was validated with a
deterministic mock evaluator, not the real OLMo checkpoints).

## Resolution of the 6 open items from `docs/phase0-findings.md`

### 1. PolyGuardPrompts Korean subset count and label balance — RESOLVED (corrected 2026-08-18)

> **Correction notice**: the numbers originally reported here contained a
> computational error, caught during a Phase 1 review pass before Phase 2
> approval. The original text claimed "945 unharmful / 754 harmful" summed
> to the full 1,725-item subset and separately claimed a "98.49% EN–KO
> agreement" as if it were an independent replication of the source paper's
> figure. Both numbers were real outputs of real code, but the second was an
> artifact of a `NaN != NaN` comparison bug, not a genuine measurement. Full
> recomputation and root-cause explanation: `docs/phase1-review.md` section 1.
> This section now states the corrected figures only.

Downloaded the dataset's single parquet file directly
(`ToxicityPrompts/PolyGuardPrompts`, `refs/convert/parquet` revision, 52.8MB)
and inspected it with pandas rather than relying on the card text.

- **Korean subset size: exactly 1,725 items** (not "~29,300/17 ≈ 1,724" —
  confirmed exact, and the true dataset total is 29,325 = 17 × 1,725, not the
  "29,300" figure reported on the card, which was evidently rounded).
- **`prompt_harm_label` value_counts(dropna=False), Korean subset: 945
  unharmful, 754 harmful, 26 None (missing).** 945 + 754 + 26 = 1,725. The
  English subset has the identical breakdown, on the identical 26 `id`s.
- **Label balance among comparable pairs is 100% (945 unharmful / 754
  harmful class balance), and EN–KO agreement on the 1,699 pairs where both
  sides have a resolved label is 100% (1,699/1,699), not 98.49%.** The
  earlier 98.49% figure was the same 1,699 count divided by the wrong
  denominator (1,725, including the 26 unresolved pairs) via a buggy
  equality comparison — see `docs/phase1-review.md` section 1 for the exact
  mechanism and the corrected calculation.
- The 26 missing-label items (same 26 `id`s in both languages) have a
  populated `prompt_label` field (20 "safe" / 6 "unsafe") but no resolved
  `prompt_harm_label`, and `prompt_harm_agreement` is null for all 26 —
  consistent with these being items where the dataset's own
  annotator-agreement process didn't converge on a harm label, not a defect
  in this project's loader.

This resolves the "does an EN–KO parallel structure actually exist" question
better than hoped: the loader doesn't need to invent a pairing heuristic, the
dataset already has one — and it turns out to be a stronger structure than
first reported (perfect agreement on every resolved pair), not a weaker one.

### 2. Korean provenance from the source paper — RESOLVED (corrected 2026-08-18)

The PolyGuard paper (arXiv 2504.04377) reports translation quality was
verified by bilingual evaluators: 99.1% of translations rated high-quality,
98.4% agreement with the original English safety labels, across the 17
languages in aggregate (not broken out per-language in what was accessible).

The Phase 1 attempt to independently reproduce this Korean-specific was
**invalid** (see correction notice in section 1 above) — the 98.49% figure
that appeared to closely match the paper's 98.4% was actually a NaN-handling
bug, not a real measurement, and the resemblance to the paper's number was
coincidental. The corrected, validly-computed figure is **100% agreement
(1,699/1,699) among pairs where both languages have a resolved label**, plus
26 pairs (1.5%) where neither language has a resolved label at all.
Conclusion: Korean items are **translated, not native**, consistent with the
paper's own methodology description, but this project's own data does not
independently corroborate the paper's specific 98.4% aggregate figure one
way or the other — the comparable subset here shows no disagreement at all,
which is different from (and cannot be used to confirm or dispute) a
17-language aggregate reported elsewhere. Treat the paper's 99.1%/98.4%
figures as the source of translation-quality provenance, and this project's
100%-on-1,699-pairs figure as a separate, narrower, correctly-computed
observation about this specific EN–KO slice.

### 3. Canonical XSTest source — RESOLVED

Fetched `xstest_prompts.csv` directly from `github.com/paul-rottger/xstest`
(root of the repo, CC-BY-4.0 confirmed in the same repo). Confirmed: **450
rows**, columns `id, prompt, type, label, focus, note`, 18 types (9 safe/9
contrast-unsafe pairs, 25 items each = 450), matching the paper's "250
safe + 200 unsafe" description exactly once summed by type. A local snapshot
is saved at `data/xstest/xstest_prompts_v2.csv` with its SHA-256 recorded in
`configs/benchmarks/registry.yaml`, so the project doesn't depend on GitHub
being reachable or unchanged at run time.

The HF mirror referenced during the Phase 0 bake-off
(`natolambert/xstest-v2-copy`) was confirmed to be a *different* artifact
(2,700 rows of prompt+completion+annotation data), not the raw prompt set —
this was caught and corrected in `docs/phase0-findings.md` before Phase 1
started, and is not used anywhere in this codebase.

### 4. KO translation subset readiness — NOT READY (as expected, now precisely scoped)

Still not ready for Phase 2 reporting, and this was never going to be
resolved by "checking" — it required doing translation work. What Phase 1
adds: a small 16-item (8 safe/unsafe pairs) draft translation, produced to
exercise `XSTestLoader` end-to-end, at
`data/xstest_ko_draft/xstest_ko_draft.jsonl`, explicitly marked
`translation_status: draft_ai_translated_pending_native_review` on every
record. Two items were evaluated and **excluded** (not force-translated)
because their over-refusal test depends on an English-only lexical
double-meaning ("terminate," "kill the lights") that doesn't survive
translation into Korean — see `data/xstest_ko_draft/excluded_items.jsonl`
and `PROVENANCE.md`. This is concrete confirmation that the exclusion risk
flagged in Phase 0 §4 is real, observed twice already in a 16-item sample,
not a hypothetical edge case.

One item (id 426, "social security number" → 주민등록번호) is flagged as a
**construct-altering localization**, not a literal translation, per Phase 0's
warning that translation itself can shift what's being measured.

The full ~100–150-item stratified subsample, and the native-speaker review
step, remain open — this is explicitly a human-in-the-loop step that
shouldn't be rubber-stamped by an AI translating its own homework. The
`XSTestLoader`'s Korean path will raise/warn if asked to load more items than
exist in the draft file, rather than silently pretend more coverage exists.

### 5. Intermediate OLMo3 MLX builds — RESOLVED (previously unconfirmed)

The Phase 0 doc flagged this as unconfirmed. Phase 1 checked directly and
found the first check method (plain HTTP status code on the HF API) was
**unreliable** — a 401 was returned for both a real-but-not-actually-gated
situation and a genuinely nonexistent test repo, i.e. status code alone
doesn't distinguish "doesn't exist" from anything else in unauthenticated
requests. Switched to comparing the fetched page's `<title>` against a known
"404 – Hugging Face" baseline, which does distinguish them reliably.

Result:
- **SFT**: MLX builds exist (`bfloat16`, `8bit`, `4bit` all confirmed).
- **DPO**: MLX builds do **not** exist (confirmed genuine 404, not gating).

Practical effect: local M4 smoke-testing with *real* quantized weights is
possible for Base, SFT, and RLVR, but not DPO. This is recorded in
`configs/models/olmo3_lineages.yaml` under `local_dev_quantized`.

### 6. HF token setup status — CHECKED, not required yet

No `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env var and no cached token file exist
on this machine. Not currently a blocker: every repo used in this project
(all four OLMo Instruct-lineage checkpoints, PolyGuardPrompts, Belebele) is
public and ungated, and Phase 1 code never required authentication. This
will matter more in Phase 2 for anonymous rate limits on repeated/bulk
downloads from a rented GPU box — recommend setting one up before Phase 2,
not before Phase 1.

---

## What was implemented (minimal evaluation core)

See the repo tree and code-path explanation in the chat response for this
turn. In one line: `configs/` (verified, human-readable) → `CheckpointRegistry`
/ dataset loaders (typed, Pydantic-validated) → `Evaluator` interface (mock +
real HF implementation, the latter untested here since this machine has no
torch/GPU) → deterministic output parsing → rule-based metrics →
`ResultRecord`/`ResultStore` (JSONL) → `ExperimentRunner` → one CLI script.

No FastAPI, no agent/tool-calling, no Docker, no CI — out of scope for this
phase per the user's explicit instruction.
