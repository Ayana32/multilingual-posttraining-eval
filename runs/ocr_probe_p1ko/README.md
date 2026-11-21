# OCR probe: vLLM image token cost (P1_ko, Qwen3-VL-4B)

Measured 2026-11-21T06:41:43Z on the Phase 4 GPU pod (A40 secure), vLLM 0.34.0 serving
`Qwen/Qwen3-VL-4B-Instruct` @ rev ebb281ec70b05090aa6165b016eac8ec08e71b17 with the v1
flags (`--max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes
--enable-lora --max-lora-rank 8`).

## Method

For each image, two `/v1/chat/completions` calls with an **identical** single user
text part and `max_tokens=1`, **no system prompt and no tools**: one with the image
attached as a base64 `image_url`, one without. The image token cost is
`prompt_tokens(with image) - prompt_tokens(without image)`.

- Text part (both calls): `Please transcribe the vendor and total from this receipt.`
- Images: 3 P1_ko arm-A receipts from `runs/phase4_manifest.jsonl` / `phase4/data/eval/images/`.

## Result

| image | w x h | sha256 (prefix) | prompt_tokens w/ image | w/o image | **image tokens** |
|---|---|---|---|---|---|
| phase4_eval_01_P1_ko.png | 560x480 | 6e1bdcad3b3a | 291 | 19 | **272** |
| phase4_eval_02_P1_ko.png | 560x480 | 69a73712340a | 291 | 19 | **272** |
| phase4_eval_03_P1_ko.png | 560x480 | ae0a274603bb | 291 | 19 | **272** |

All three receipts render to the same dimensions, so vLLM encodes each as the same
number of image tokens. **Each P1_ko receipt image costs 272 vLLM tokens.**
