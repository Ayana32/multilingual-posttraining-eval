#!/usr/bin/env python3
"""Phase 4 defensive LoRA SFT (docs/phase4-prereg.md section 4).

transformers 5.22.2 `Trainer` + peft 0.21.3. NO TRL. Every hyperparameter is
read from phase4/train_config.yaml. The assistant loss mask is imported from
phase4/checks.py (assistant_token_labels) and NOT reimplemented here.

GPU day: `python phase4/train.py` trains the adapter and saves the final
checkpoint. build_examples() and the collator are importable for the CPU dry-run
test (tests/test_phase4_train_dryrun.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4"))
from checks import assistant_token_labels  # single source of the loss mask

CONFIG_PATH = REPO / "phase4" / "train_config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_examples(cfg: dict, tok, tools, limit: int | None = None) -> list[dict]:
    """Tokenizes the training conversations into {input_ids, labels} using the
    shared assistant mask. Asserts no example exceeds max_seq_len (prereg: no
    truncation)."""
    train_file = REPO / cfg["data"]["train_file"]
    max_len = cfg["train"]["max_seq_len"]
    out = []
    for i, line in enumerate(open(train_file, encoding="utf-8")):
        if not line.strip():
            continue
        if limit is not None and len(out) >= limit:
            break
        ex = json.loads(line)
        ids, labels, _spans = assistant_token_labels(ex["messages"], tok, tools)
        assert len(ids) <= max_len, f"example {ex.get('receipt_id')} length {len(ids)} > max_seq_len {max_len}"
        assert len(ids) == len(labels)
        out.append({"input_ids": ids, "labels": labels})
    return out


class MaskedCollator:
    """Right-pads input_ids with pad_token_id and labels with -100; builds
    attention_mask. No TRL, no dependence on the model."""
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        m = max(len(b["input_ids"]) for b in batch)
        ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"]); pad = m - n
            ids.append(b["input_ids"] + [self.pad] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def count_lora_modules(model) -> tuple[int, int]:
    """(matched, vision): modules PEFT wrapped with a LoRA adapter (carry lora_A
    and lora_B), and how many of those are vision-tower modules. Same rule as
    scripts/make_dummy_lora.py; the v1 dummy LoRA had (144, 0)."""
    matched = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B")]
    vision = [n for n in matched if "visual" in n.lower() or "vision" in n.lower()]
    return len(matched), len(vision)


def build_training_args(cfg: dict, output_dir=None):
    """The exact TrainingArguments phase4/train.py trains with, built from the
    frozen config. Extracted so a CPU test can assert the pinned transformers
    accepts every argument (tests/test_phase4_trainingargs_build.py). main()
    builds its args through this same function, so the two cannot drift."""
    from transformers import TrainingArguments
    t = cfg["train"]
    return TrainingArguments(
        output_dir=str(output_dir if output_dir is not None else REPO / "runs" / "phase4_adapter"),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=cfg["optim"]["learning_rate"],
        lr_scheduler_type=cfg["optim"]["lr_scheduler"],
        warmup_steps=3,  # transformers 5.22.2 dropped warmup_ratio; prereg 5% warmup = ceil(0.05*50)=3 of 50 steps (docs/v1-changes.md)
        weight_decay=cfg["optim"]["weight_decay"],
        optim="adamw_torch",
        bf16=t["bf16"],
        seed=t["seed"],
        save_strategy="no",       # final checkpoint only; no selection (prereg section 4)
        logging_steps=10,
        report_to=[],
    )


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText, Trainer, set_seed
    from peft import LoraConfig, get_peft_model

    cfg = load_config()
    assert cfg["pins"]["training_library"] == "transformers 5.22.2 Trainer + peft 0.21.3", \
        "training_library must be recorded in the config before training (prereg section 4)"
    tools = json.loads((REPO / cfg["data"]["tools"]).read_text())
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["base_model_revision"])

    model = AutoModelForImageTextToText.from_pretrained(
        cfg["base_model"], revision=cfg["base_model_revision"], torch_dtype=torch.bfloat16)
    lc = cfg["lora"]
    set_seed(cfg["train"]["seed"])            # seed BEFORE creating the adapter (prereg section 4)
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
        bias="none", task_type=lc["task_type"], target_modules=lc["target_modules"]))
    n_lora, n_vision = count_lora_modules(model)
    assert n_lora == 144 and n_vision == 0, \
        f"LoRA module count ({n_lora}, vision {n_vision}) != v1 dummy LoRA (144, 0); refusing to train"

    examples = build_examples(cfg, tok, tools)
    args = build_training_args(cfg)
    trainer = Trainer(model=model, args=args, train_dataset=examples,
                      data_collator=MaskedCollator(tok.pad_token_id or tok.eos_token_id))
    # Pre-training: confirm the planned optimizer-step count from the real dataloader
    # (prereg 5% warmup is implemented as warmup_steps=3, i.e. 5% of these 50 steps).
    import math
    _dl = trainer.get_train_dataloader()
    _steps = math.ceil(len(_dl) / args.gradient_accumulation_steps) * int(args.num_train_epochs)
    print(f"[train] planned optimizer steps = {_steps} "
          f"({len(_dl)} batches/epoch, grad_accum {args.gradient_accumulation_steps}, epochs {int(args.num_train_epochs)})")
    assert _steps == 50, f"expected 50 optimizer steps (warmup_steps=3 = 5% of 50), got {_steps}; stopping"
    trainer.train()
    trainer.save_model(str(REPO / "runs" / "phase4_adapter"))
    print("saved final adapter to runs/phase4_adapter")


if __name__ == "__main__":
    main()
