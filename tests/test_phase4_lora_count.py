"""phase4/train.py::count_lora_modules counts PEFT-wrapped modules (lora_A+lora_B)
and flags vision-tower matches. train.py asserts (144, 0) -- the v1 dummy LoRA --
after get_peft_model; this tests the counting rule without the real model."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
import train  # noqa: E402  (count_lora_modules; no torch/peft imported at module load)


def _fake_model():
    import torch.nn as nn

    class LoraLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Linear(4, 2, bias=False)
            self.lora_B = nn.Linear(2, 4, bias=False)

    m = nn.Module()
    m.add_module("language_model_layers_0_self_attn_q_proj", LoraLike())
    m.add_module("language_model_layers_0_self_attn_k_proj", LoraLike())
    m.add_module("plain", nn.Linear(4, 4))          # not wrapped
    return m, LoraLike


def test_counts_lora_modules_zero_vision():
    m, _ = _fake_model()
    n, vision = train.count_lora_modules(m)
    assert n == 2 and vision == 0


def test_flags_vision_modules():
    m, LoraLike = _fake_model()
    m.add_module("visual_blocks_0_attn_qkv", LoraLike())    # a vision-tower match
    n, vision = train.count_lora_modules(m)
    assert n == 3 and vision == 1   # train.py would refuse to train on vision>0
