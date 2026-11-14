# Phase 3 fonts

Both fonts are pinned to the same commit of the `google/fonts` repository
(`eb20e9cf19d97cf579fa3867d729d4222d6c59e1`), SIL Open Font License 1.1
(OFL), and used as variable fonts set to the `Regular` named instance at
render time (`ImageFont.set_variation_by_name("Regular")`).

## NotoSans-Variable.ttf

- **Source:** https://raw.githubusercontent.com/google/fonts/eb20e9cf19d97cf579fa3867d729d4222d6c59e1/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf
- **Version (name table, nameID 5):** Version 2.015
- **License:** SIL Open Font License 1.1 (see `OFL-NotoSans.txt`)
- **SHA-256:** `bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d`

## NotoSansKR-Variable.ttf

- **Source:** https://raw.githubusercontent.com/google/fonts/eb20e9cf19d97cf579fa3867d729d4222d6c59e1/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf
- **Version (name table, nameID 5):** Version 2.004-H2;hotconv 1.0.118;makeotfexe 2.5.65603
- **License:** SIL Open Font License 1.1 (see `OFL-NotoSansKR.txt`)
- **SHA-256:** `194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252`

Note: this font's own name-table family name (nameID 1) reads "Noto Sans KR Thin"
at the default instance before a named instance is selected; the file is the
same `NotoSansKR[wght].ttf` variable font referenced in
`docs/phase3-visual-injection.md` section 4.2, used at the `Regular` instance.

## NanumGothic-Regular.ttf (added for Phase 3b arm B)

Added for `docs/phase3b-replication.md` Section 7 (arm B, font sensitivity): a
single font file covering both Latin and Hangul, used for both `P1_en` and
`P1_ko` footers instead of switching font by script. Unlike the two Noto fonts,
this is a **static** font (no `fvar` table), so `set_variation_by_name` is NOT
called on it (see Section 7); it is loaded with
`ImageFont.truetype(..., 16, layout_engine=ImageFont.Layout.BASIC)` and nothing
more.

- **Source:** https://raw.githubusercontent.com/google/fonts/eb20e9cf19d97cf579fa3867d729d4222d6c59e1/ofl/nanumgothic/NanumGothic-Regular.ttf
- **Commit:** `eb20e9cf19d97cf579fa3867d729d4222d6c59e1` (the same `google/fonts`
  commit as the two Noto fonts above)
- **Version (name table, nameID 5):** Version 3.020;PS 1;hotconv 1.0.57;makeotf.lib2.0.21895
- **License:** SIL Open Font License 1.1 (see `OFL-NanumGothic.txt`)
- **SHA-256:** `76f45ef4a6bcff344c837c95a7dcc26e017e38b5846d5ae0cdcb5b86be2e2d31`
- Covers Latin (`A`/`a`) and the full Hangul syllable range bounds
  (U+AC00, U+D7A3); verified, not assumed. Static: no `fvar` table.

## License files

- `OFL-NotoSans.txt` -- SHA-256: `cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a`
- `OFL-NotoSansKR.txt` -- SHA-256: `1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9`
- `OFL-NanumGothic.txt` -- SHA-256: `eeacf16032901d0ed0456876ec77b8f0fda6b3fecec7d972f8543eb602e6c30f`

## Provenance note

Commit A did not specify where these font files themselves should come from;
see `docs/phase3-deviations.md` for that clarification, logged before any run.
