#!/usr/bin/env python3
"""Build a zero-effect LoRA adapter (docs/v1-spec.md Section 5, check 4).

Produces a PEFT LoRA adapter whose lora_B weights are all zero, so the
adapter is mathematically identity: y = W0 x + (B A) x = W0 x. Serving it
in vLLM must therefore return exactly the base model's behaviour. Check 4
only asks whether "a dummy LoRA adapter loads, and one image + tools
request with that adapter selected returns a response" -- a zero-effect
adapter tests the *serving path* (adapter load + routing) without changing
any output, so a check-4 failure is unambiguously a serving problem, not a
model one.

This runs on the pod, where the base model and peft/torch are available.
Locally (no GPU, peft/torch often absent) use:

    --dry-run                 validate args and print the planned config
                              (no base-model download, no heavy imports if
                              peft is missing)
    --base-model <tiny-id>    point at a tiny random HF model to exercise
                              the real build path cheaply

The base model and its revision must be pinned (a 40-hex commit), matching
the eval's model-revision discipline: the adapter is built against one
exact base snapshot.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")

# Attention + MLP projections: the standard LoRA targets for a decoder /
# VL decoder stack. lora_B starts at zero in PEFT, so with this left at its
# init the adapter is already identity; we additionally hard-zero lora_B
# after construction as a guarantee that does not depend on PEFT internals.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _planned_config(args) -> dict:
    return {
        "base_model": args.base_model,
        "base_model_revision": args.revision,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": args.r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": args.target_modules,
        "lora_B": "all zeros (zero-effect / identity adapter)",
        "out_dir": str(Path(args.out).resolve()),
    }


def _build_lora_config(args):
    from peft import LoraConfig  # imported lazily: only on the pod / real build

    return LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )


def _zero_lora_B(model) -> int:
    """Sets every lora_B tensor to zero in place; returns how many it zeroed.
    (PEFT already inits lora_B to zero, so this is a belt-and-braces check
    that the saved adapter is exactly identity.)"""
    import torch

    zeroed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
                zeroed += 1
    return zeroed


def build(args) -> dict:
    """The real build: load the base model at its pinned revision, attach a
    LoRA adapter, zero lora_B, and save the adapter folder. Runs on the pod."""
    import torch  # noqa: F401
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    config = _build_lora_config(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision, torch_dtype="auto",
    )
    peft_model = get_peft_model(model, config)
    zeroed = _zero_lora_B(peft_model)
    if zeroed == 0:
        raise SystemExit("no lora_B parameters found -- target_modules matched nothing; refusing to save")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out))

    # Record provenance alongside the adapter (what base + revision it is for).
    provenance = _planned_config(args)
    provenance["lora_B_tensors_zeroed"] = zeroed
    (out / "dummy_lora_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a zero-effect LoRA adapter (docs/v1-spec.md Section 5).")
    parser.add_argument("--base-model", required=True, help="base model id the adapter is built for")
    parser.add_argument("--revision", required=True, help="40-hex commit revision of the base model (pinned)")
    parser.add_argument("--out", required=True, help="output folder for the saved adapter")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate args and print the planned config; no base-model download")
    args = parser.parse_args(argv)

    if not _COMMIT_HASH.fullmatch(args.revision):
        parser.error(f"--revision must be a 40-character hex commit hash, not {args.revision!r}")

    planned = _planned_config(args)

    if args.dry_run:
        # Construct the LoraConfig too, if peft is importable, to validate it
        # for real; otherwise just print the intended parameters.
        try:
            cfg = _build_lora_config(args)
            planned["lora_config_repr"] = repr(cfg)
            planned["peft_available"] = True
        except ImportError:
            planned["peft_available"] = False
            planned["note"] = "peft not installed here; printed intended config only (run the build on the pod)"
        print(json.dumps(planned, indent=2))
        return 0

    result = build(args)
    print(json.dumps(result, indent=2))
    print(f"saved zero-effect LoRA adapter to {result['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
