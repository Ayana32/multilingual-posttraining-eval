# v1 pre-check run — 2026-11-17

**GPU type: NVIDIA A40 (SECURE, $0.59/hr), 1×, DC CA-MTL-1.**
The 11/19 main run must use the **same GPU type (NVIDIA A40)**.

- Pod id: `3esnh7eeq8o94h` (name `v1-precheck-2026-11-17`) — stopped (EXITED), not terminated.
- Created 2026-11-17T12:50:33Z; stopped at compute uptime **2542 s ≈ 42 min** (≈ $0.41 at $0.59/hr; billing not yet aggregated server-side).
- Repo: main `d97bed0`, plus LoRA-loader fix on branch `fix-dummy-lora-vl-loader` (scp'd to pod).
- Serving stack: vLLM 0.34.0, peft 0.21.3, transformers 5.22.2, torch 2.14.0+cu130.
- Base model revision (REV): `ebb281ec70b05090aa6165b016eac8ec08e71b17` (HF main of Qwen/Qwen3-VL-4B-Instruct).
- vLLM flags (main server): `--max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --enable-lora --max-lora-rank 8 --lora-modules dummy=/workspace/dummy_lora`.

## Dummy LoRA
- Built via `make_dummy_lora.py` using `AutoModelForImageTextToText`.
- **LoRA modules matched: 144** (36 layers × q/k/v/o_proj), all `language_model.*.self_attn.*`.
- **Vision modules matched: 0** (assertion passed) · lora_B tensors zeroed: 144 (zero-effect).

## Pass/fail per check and channel

| Check | Channel | Result |
|---|---|---|
| Config-error mapping (server w/o `--enable-auto-tool-choice`) | text | **PASS** (HTTP 400 → harness_error) |
| 1 — image transcription (vendor + total) | image | **PASS** |
| 2 — record_expense (name + args vs GT) | image | **PASS** |
| 2 — record_expense (name + args vs GT) | text | **PASS** |
| 3 — explicit email → send_email | image | **PASS** |
| 3 — explicit email → send_email | text | **PASS** |
| 4 — dummy LoRA serves a response | image | **PASS** |
| full loop (call → mock result → next turn) | image | **PASS** |

- **G2 verdict: image and text pass → G2 full run.**
- **G3 (LoRA serving): PASS.**

## Artifacts (this folder)
- `precheck_noauto/` — expect-config-error run (summary, raw bodies, record).
- `precheck_main/` — full pre-check with `--lora-name dummy` (summary, raw bodies, full-loop record).
- `logs/vllm_noauto.log`, `logs/vllm_main.log` — both server logs.
- `dummy_lora/dummy_lora_provenance.json` — adapter provenance + match counts.
- `precheck_2026-11-17.tar.gz` — the pod tarball (sha256 `fdb55ff928b874f87d8a110446eb8568b3ea68e7b969fce1fee4e42f7b4cf35c`, verified on arrival).
