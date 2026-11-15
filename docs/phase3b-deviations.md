# Phase 3b Deviations Log

Per `docs/phase3b-replication.md` ("Deviations rule"): every change after the
`phase3b-prereg` tag is logged here with the date, the change, the reason, and the commit
hash from which it applies, written before the change is used in any run it would affect.

## 2026-11-15 -- G3 (b) and (c) reviewed in one local-viewer pass

**Change.** G3 (b) (the manual contact-sheet checklist, all 440 images) and G3 (c) (the
manual line-by-line comparison: all 400 footers exhaustively + 40 sampled bodies) are
performed in a single pass in a local viewer (`phase3b/g3_review.py`), instead of the
Section-10 workflow of skimming a contact-sheet grid for (b) and then separately reading
line-by-line for (c). Per cell (arm, font, variant), the viewer stacks full-resolution
crops of each image's footer and of the TOTAL-to-footer boundary, with that image's
reference transcript footer lines shown on top of each crop; the 40 sampled body images are
shown as a body crop beside their transcript body lines. Crop boxes are the exact draw
bounding boxes recorded by G3 (a) (`report/phase3b/g3a_results.json`, `images[].draws[]`,
added for this), plus a small fixed margin.

**What does not change.** Every check listed in Section 10 (b) and (c) is still entered by
hand for every image -- the same checklist (Korean renders; "Receipt copy" and the address
each on one line; the extra line not cut off/overlapping; body-to-footer spacing; every
footer line line-by-line; the 40 sampled bodies line-by-line). There is no bulk-pass and no
automatic judgment: a value is recorded only when a human enters it, a fail requires a note,
and the n/a cells (per the Section-10 checklist, already in
`report/phase3b/g3_manual_review.csv`) stay locked. The contact sheets
(`phase3b/make_contact_sheets.py`) are still generated and committed. No image, transcript,
or pre-registered measure is affected.

**Reason.** A 280/80/80-cell contact-sheet skim followed by a separate 400-footer read is
slower and more error-prone than looking at each image's actual rendered footer and the
body-to-footer boundary at full resolution, with the reference transcript lines beside them,
once. Showing the exact rendered region a check is about -- rather than a thumbnail in a
grid -- makes the per-image checklist entries more reliable, which is the point of the gate.

**Applies from commit:** this commit (recorded before the viewer is used for any review).
