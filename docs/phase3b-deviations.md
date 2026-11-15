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

## 2026-11-15 -- Eval pod Pillow/FreeType below the pinned render environment

**Change.** The R_eval run was executed on a pod whose imaging stack was Pillow 11.0.0 /
FreeType 2.13.2, below the render environment pinned in `docs/phase3b-replication.md`
(Section 7) and the runbook -- Pillow 11.1.0 / FreeType 2.13.3, the pair recorded in
`phase3b/data/environment.json` that rendered the committed 440 images. All 28
`report/phase3b_eval/*/environment.json` records report `"pillow_version": "11.0.0"`. The
eval `environment.json` does not capture a FreeType version; the 2.13.2 figure is the
FreeType version printed by the on-pod pre-flight (runbook step 5 output).

**Why this is logged here.** The runbook pre-flight ("Image hashes", step 5) re-renders the
committed images in memory on the pod via `phase3b/g3_check.py`
(`make_receipts.render_receipt`) and asserts byte-for-byte identity against the committed
PNGs. Both the pre-registration and the runbook gate the meaning of that check on the
render-environment match: "if Phase 3b is actually run on a different Pillow/FreeType pair,
the determinism check against arm A's images is not meaningful ... and that mismatch is
logged in `docs/phase3b-deviations.md`, not silently ignored" (replication.md, Section 7);
"If the pod's Pillow/FreeType differ from the committed render environment ... the pixel
check is not meaningful and the mismatch is logged in `docs/phase3b-deviations.md`"
(runbook). The pod's pair differs, so this entry is that log.

**When it was seen, and why it is logged now.** The mismatch was visible in the on-pod
pre-flight itself: runbook step 5 printed Pillow 11.0.0 | FreeType 2.13.2, against the
committed render environment's 11.1.0 / 2.13.3. That session judged the difference benign
(the pre-flight's pixel check still passed -- see below) and did not log it. It is logged
here now, before any write-up uses the R_eval results.

**Scope -- what is and is not affected.** The pins apply to *rendering* only. The
pre-registered measures are computed from the committed, SHA256-verified image bytes
(`run.py` feeds `row["image_sha256"]`; `report/phase3b_eval/SHA256SUMS` covers the run
outputs), and the merge/eval gate pins only the inference stack (`transformers==4.57.1`,
`torch==2.8.0+cu128` -- `score_eval.py` `PINNED_VERSIONS`), not Pillow. No copy rate,
McNemar table, bootstrap CI, d_i, or transcription CER depends on the pod's Pillow/FreeType
version. The only check these versions feed is the on-pod G3(a) pixel-reproduction
pre-flight -- and that pre-flight reproduced all 440 committed images byte-for-byte on the
pod's 11.0.0 / 2.13.2 (`all_committed_pixels_match` true), so the images are confirmed
identical to the committed set despite the version gap.

**Arm B byte-identity is unaffected for a separate reason.** Arm B's byte-identity claim --
NotoSansKR x `P1_ko` renders the same image bytes as arm A's `P1_ko` -- was established on
the render host at image-generation time (Pillow 11.1.0 / FreeType 2.13.3), not on the eval
pod. The eval pod consumes those already-committed image bytes, so this version gap does not
touch the arm B byte-identity.

**Applies from commit:** c6546fc (the R_eval results commit).
