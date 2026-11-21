"""CPU, no-network guard: the TrainingArguments that phase4/train.py builds from
phase4/train_config.yaml must be accepted by the pinned transformers (5.22.2).

This is the construction the CPU dry-run test (test_phase4_train_dryrun.py) does
NOT exercise -- it only builds the dataset and loss mask. That gap let the
`warmup_ratio` incompatibility reach GPU day (docs/v1-changes.md, 2026-11-21):
transformers 5.22.2 removed `warmup_ratio`, so the frozen call raised at
construction. This test builds the real args through phase4.train.build_training_args
(the same helper main() uses), so if the pinned transformers rejects ANY argument
the call raises and this test fails. No model, no tokenizer, no network.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4"))

# Not a network test: TrainingArguments construction needs no download. Skip only
# if transformers is not installed at all (CI installs the pinned 5.22.2).
pytest.importorskip("transformers")


def test_training_args_build_from_config_is_accepted(tmp_path):
    from transformers import TrainingArguments
    import train  # phase4/train.py

    cfg = train.load_config()

    # Exactly the construction phase4/train.py main() performs. If transformers
    # rejects any argument (as 5.22.2 did for warmup_ratio), this raises and fails.
    args = train.build_training_args(cfg, output_dir=tmp_path)

    assert isinstance(args, TrainingArguments)
    # prereg 5% warmup, expressed as warmup_steps for transformers 5.22.2
    assert args.warmup_steps == 3
    # config values round-tripped into the real object
    assert int(args.num_train_epochs) == cfg["train"]["num_train_epochs"]
    assert args.gradient_accumulation_steps == cfg["train"]["gradient_accumulation_steps"]
    assert args.per_device_train_batch_size == cfg["train"]["per_device_train_batch_size"]
