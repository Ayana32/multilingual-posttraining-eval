# Phase 3 Deviations Log

Per `docs/phase3-visual-injection.md` Section 12: every change after Commit A is logged
here with the date, the change, the reason, and the commit hash from which it applies.
Both entries below are logged before any run (no `R_tune` or `R_eval` run has happened
yet), so they fall under "before any `R_eval` run" in Section 12.

## 2026-11-05: receipt body language fixed to English

- **Date:** 2026-11-05
- **Change:** Receipt body language is English for all receipts (vendor names, items,
  currency GBP). Only the extra line varies by language.
- **Reason:** Commit A did not specify the body language; fixing it keeps payload
  language as the only language variable. Logged before any run.
- **Applies from:** the commit that adds this entry (see `git log -- docs/phase3-deviations.md`).

## 2026-11-05: font source files

- **Date:** 2026-11-05
- **Change:** Font source files were not specified in Commit A; the files used are
  bundled and documented in `phase3/fonts/README.md`.
- **Reason:** Noto Sans and Noto Sans KR were not available locally; both were obtained
  from the `google/fonts` repository, pinned to a specific commit, and bundled for
  reproducibility. Logged before any run.
- **Applies from:** the commit that adds this entry (see `git log -- docs/phase3-deviations.md`).

## Limitations (additions to Section 11)

- Korean payloads appear inside English receipts.
- Latin text inside Korean payload lines (address, subject) is rendered with Noto Sans KR
  glyphs, while English lines use Noto Sans.
- EN and KO payloads wrap to different line counts.
