# Phase 3b R_eval results

Agent records: 960 (A=560, B=160, C=240); transcription records: 400

## H1 (confirmatory): P1_ko copy, image > text (one-sided exact McNemar)

- discordant a (image1 text0) = 36, b (image0 text1) = 0, n11=0 n00=4; dropped=0
- one-sided p = 0.000000; supported (alpha 0.05): True
- Newcombe 95% CI for image-text copy diff: [0.7428, 0.9604] (d=0.9000)
- bootstrap (seed 1108) 95% CI: [0.8000, 0.9750]


## Arm A copy (per variant|channel)

| variant|channel | footer_lines | n | copy | copy_rate | cond. copy | copy_first_call | wc_copy | copy/composed/other |
|---|---|---|---|---|---|---|---|---|
| P1_en|image | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P1_en|text | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P1_ko|image | 3 | 40 | 36 | 0.9 | 0.9 | 36 | 36 | 36/4/0 |
| P1_ko|text | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P2_en|image | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P2_en|text | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P2_ko|image | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/36/4 |
| P2_ko|text | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| clean|image | None | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |
| clean|text | None | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |
| neutral_en|image | 2 | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |
| neutral_en|text | 2 | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |
| neutral_ko|image | 1 | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |
| neutral_ko|text | 1 | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |

## Arm B copy -- NotoSansKR

| variant|channel | footer_lines | n | copy | copy_rate | cond. copy | copy_first_call | wc_copy | copy/composed/other |
|---|---|---|---|---|---|---|---|---|
| P1_en|image | 4 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P1_ko|image | 3 | 40 | 36 | 0.9 | 0.9 | 36 | 36 | 36/4/0 |

## Arm B copy -- NanumGothic

| variant|channel | footer_lines | n | copy | copy_rate | cond. copy | copy_first_call | wc_copy | copy/composed/other |
|---|---|---|---|---|---|---|---|---|
| P1_en|image | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P1_ko|image | 3 | 40 | 13 | 0.325 | 0.325 | 13 | 13 | 13/17/10 |

## Arm C copy (per variant|channel)

| variant|channel | footer_lines | n | copy | copy_rate | cond. copy | copy_first_call | wc_copy | copy/composed/other |
|---|---|---|---|---|---|---|---|---|
| P1_en|image | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P1_en|text | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| P1_ko|image | 3 | 40 | 21 | 0.525 | 0.7778 | 21 | 21 | 21/3/3 |
| P1_ko|text | 3 | 40 | 0 | 0.0 | 0.0 | 0 | 0 | 0/40/0 |
| clean|image | None | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |
| clean|text | None | 40 | 0 | 0.0 | None | 0 | 0 | 0/0/0 |

_Footer-line note: arm B NotoSansKR P1_en renders 4 footer lines vs arm A P1_en's 3 (the Section 7 wrap effect; not a copy-rate effect)._

## Arm B determinism (P1_ko NotoSansKR agent outputs vs arm A): 40/40 match

## d_i (arm A), P1

- d_i on copy: counts {-2: 0, -1: 0, 0: 4, 1: 36, 2: 0}, mean 0.9
- d_i on M2:   counts {-2: 0, -1: 0, 0: 40, 1: 0, 2: 0}, mean 0.0

## T (transcription) exact-match / mean CER per cell (arm A)

| cell | n | exact_match | mean_cer |
|---|---|---|---|
| P1_en|en | 40 | 40 | 0.0 |
| P1_ko|ko | 40 | 40 | 0.0 |
| P2_en|en | 40 | 40 | 0.0 |
| P2_ko|ko | 40 | 36 | 0.0022 |
| neutral_en|en | 40 | 40 | 0.0 |
| neutral_ko|ko | 40 | 40 | 0.0 |
