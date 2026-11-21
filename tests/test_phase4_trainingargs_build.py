"""CPU, no-network guard: every argument that phase4/train.py passes to
TrainingArguments from phase4/train_config.yaml must be accepted by the pinned
transformers (5.22.2).

This is the construction the CPU dry-run test (test_phase4_train_dryrun.py) does
NOT exercise -- it only builds the dataset and loss mask. That gap let the
`warmup_ratio` incompatibility reach GPU day (docs/v1-changes.md, 2026-11-21):
transformers 5.22.2 removed `warmup_ratio`, so the frozen call raised
`TypeError: ... unexpected keyword argument 'warmup_ratio'` at construction.

The args are built through phase4.train.build_training_args, the same helper main()
uses, so the two cannot drift. A rejected argument (an unknown/removed keyword) is a
TypeError raised at signature binding, before __post_init__; this test fails on that.
A GPU-only value check (bf16=True needs a GPU) is a ValueError raised later in
__post_init__, after every argument name has been accepted; on a CPU-only runner that
is an environment gate, not an argument rejection, so the guard still holds and the
test tolerates it. No model, no tokenizer, no network.
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

    try:
        # Exactly the construction phase4/train.py main() performs.
        args = train.build_training_args(cfg, output_dir=tmp_path)
    except TypeError as e:
        # An argument was rejected (as 5.22.2 did for warmup_ratio). This is the
        # regression this test exists to catch.
        pytest.fail(f"transformers rejected a TrainingArguments argument: {e}")
    except ValueError as e:
        # Every argument name was accepted (we reached __post_init__). On a CPU-only
        # runner bf16=True is refused for lack of a GPU; that is an environment gate,
        # not an argument rejection, so the guard still holds.
        assert ("bf16" in str(e)) or ("use_cpu" in str(e)), f"unexpected ValueError: {e}"
        return

    # Constructed fully (e.g., on a GPU runner): check the values round-tripped.
    assert isinstance(args, TrainingArguments)
    assert args.warmup_steps == 3  # prereg 5% warmup, expressed for 5.22.2
    assert int(args.num_train_epochs) == cfg["train"]["num_train_epochs"]
    assert args.gradient_accumulation_steps == cfg["train"]["gradient_accumulation_steps"]
    assert args.per_device_train_batch_size == cfg["train"]["per_device_train_batch_size"]
