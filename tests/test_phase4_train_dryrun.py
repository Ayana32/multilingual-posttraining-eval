"""CPU dry-run for phase4/train.py on 2 examples. Confirms the labels are -100
everywhere except the assistant spans (and equal the input ids inside them), then
runs a tiny mocked forward+backward through the real collator on CPU to exercise
the masking end to end. Needs the Qwen3-VL-4B tokenizer -> marked `network`.

The assistant-mask function is the one imported from phase4/checks.py (train.py
imports it; this test does not reimplement it)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase4"))
import checks  # noqa: E402  (assistant_token_labels)
import train   # noqa: E402  (build_examples, MaskedCollator, load_config)

REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


@pytest.mark.network
def test_labels_minus100_outside_assistant_spans_and_tiny_forward():
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", revision=REV)
    tools = json.loads((REPO / "phase3" / "fixtures" / "tools.json").read_text())
    cfg = train.load_config()

    examples = train.build_examples(cfg, tok, tools, limit=2)
    assert len(examples) == 2

    raw = [json.loads(l) for l in open(REPO / cfg["data"]["train_file"], encoding="utf-8")][:2]
    for ex, src in zip(examples, raw):
        ids, labels, spans = checks.assistant_token_labels(src["messages"], tok, tools)
        assert ex["input_ids"] == ids and ex["labels"] == labels
        in_span = set()
        for a, b in spans:
            in_span.update(range(a, b))
        assert spans and all(b > a for a, b in spans)
        for j, lab in enumerate(labels):
            if j in in_span:
                assert lab == ids[j], f"assistant token {j} must be in loss"
            else:
                assert lab == -100, f"non-assistant token {j} must be masked"
        # gold tool-call argument values are inside the loss region
        loss_text = tok.decode([t for t, l in zip(ids, labels) if l != -100])
        for m in src["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for v in m["tool_calls"][0]["function"]["arguments"].values():
                    assert str(v) in loss_text

    # tiny mocked forward through the real collator (CPU): loss uses ignore_index=-100
    collate = train.MaskedCollator(tok.pad_token_id or tok.eos_token_id)
    batch = collate(examples)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    vocab = int(batch["input_ids"].max().item()) + 1
    emb = torch.nn.Embedding(vocab, 8)
    head = torch.nn.Linear(8, vocab)
    logits = head(emb(batch["input_ids"]))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab), batch["labels"].view(-1), ignore_index=-100)
    loss.backward()
    assert torch.isfinite(loss) and (batch["labels"] != -100).sum() > 0
