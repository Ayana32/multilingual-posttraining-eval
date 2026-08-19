# XSTest Korean draft translation — provenance

**Status: DRAFT. NOT approved for any Phase 2 reported result.**

This is a 16-item (8 safe/unsafe pair) subsample, translated to exercise the
`XSTestLoader` end-to-end during Phase 1 development. It is **not** the
~100–150-item stratified subsample the protocol in `docs/phase0-findings.md`
§4 calls for, and it has **not** received the native-speaker review that
protocol requires before any translated item can be trusted.

- **Translator**: Claude (Anthropic), acting as the implementing agent for this
  project — an AI draft, not a professional human translator.
- **Date**: 2026-08-18
- **Source**: `data/xstest/xstest_prompts_v2.csv`, itself a snapshot of
  `https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv`
  fetched 2026-08-18 (see checksum/fetch note in `configs/benchmarks/registry.yaml`).
- **Method**: each item was translated individually with the item's intended
  safe/unsafe framing in view (not translated blind, sentence-by-sentence,
  without knowing which construct it's supposed to test), per the Phase 0
  protocol's requirement to preserve behavioural framing, not just literal
  meaning.
- **Per-item confidence**: recorded in the `translation_note` field of each
  record in `xstest_ko_draft.jsonl`. Several are flagged as needing a native
  speaker's naturalness judgment even where the translation is defensible.
- **Exclusions**: 2 items from the same categories were evaluated and
  excluded rather than force-translated — see `excluded_items.jsonl`. Both
  exclusions are cases where the English item's safety-test logic depends on
  an English-specific lexical double-meaning ("terminate," "kill the lights")
  that has no equivalent double-meaning in Korean, so a literal translation
  would stop testing over-refusal and start testing an ordinary, unambiguous
  question instead.
- **Construct-altering substitution flagged**: item 426 ("social security
  number" → 주민등록번호, Korean Resident Registration Number) is a
  functional-equivalent localization, not a literal translation, and is
  flagged in its `translation_note` for reviewer attention — see
  `docs/phase0-findings.md` §4's warning about translation altering the
  construct being measured.

## Before this data can be used in Phase 2

1. A native Korean speaker must review all 16 items (and any items added
   later to reach the full protocol subsample) for naturalness and construct
   preservation, per `docs/phase0-findings.md` §4.
2. The full stratified subsample (~100–150 items across all 18 XSTest types)
   still needs to be produced — this draft only covers 8 of the 9 safe/unsafe
   category pairs, 1 pair each, as a pipeline smoke-test fixture.
3. `translation_status` on each record should be updated from
   `draft_ai_translated_pending_native_review` to `human_reviewed` (or
   corrected/re-translated as needed) before use.
