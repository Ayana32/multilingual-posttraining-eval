"""Reproducibility check for the Phase 2A pilot's 50-item sample.

Pins the exact class balance observed and recorded in
docs/phase2a-pilot.md/configs/experiments/pilot_base_en_polyguard.yaml, so a
future dependency bump (e.g. a change in Python's random module behaviour,
astronomically unlikely but the whole point of a reproducibility test is not
trusting "unlikely") can't silently change what the pilot actually samples
without a test noticing.

Split across two fixture tiers:
- Tests about the *sampling mechanism itself* (determinism, subset
  relationship, label pass-through) run against the tiny local
  `tiny_polyguard_parquet` fixture -- no network, safe for CI.
- Tests that pin an actual documented property of the *real* full
  PolyGuardPrompts dataset (exact row count, exact null-label rows, the
  real pilot's real 28/22 split) genuinely need the real cached parquet and
  are marked `@pytest.mark.network` -- CI skips these via
  `pytest -m "not network"`; local/pod runs still cover them.
"""

from collections import Counter

import pandas as pd
import pytest

from mpe.datasets.polyguard import PolyGuardPromptsLoader

PILOT_SEED = 42
PILOT_LIMIT = 50


def test_pilot_sample_is_reproducible_across_calls(tiny_polyguard_parquet):
    """Determinism of the sampling mechanism -- doesn't depend on which
    dataset is behind it, so the tiny fixture (3 EN rows) is sufficient."""
    loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
    first = [i.parallel_item_id for i in loader.load("en", limit=2, seed=PILOT_SEED)]
    second = [i.parallel_item_id for i in loader.load("en", limit=2, seed=PILOT_SEED)]
    assert first == second


def test_pilot_sample_contains_both_classes(tiny_polyguard_parquet):
    """load() correctly surfaces both harmful and unharmful labels when
    present in the source data. Uses limit=3 (the tiny fixture's full EN
    row count) rather than a sub-sample: with only 1 harmful row out of 3,
    a smaller limit would make "both classes present" depend on seed luck
    rather than genuinely testing this property. The sampling mechanism's
    own correctness (determinism, subset relationship) is covered by the
    other two mechanism-level tests in this file."""
    loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
    items = loader.load("en", limit=3, seed=PILOT_SEED)
    counts = Counter(i.expected_label for i in items)
    assert counts["harmful"] > 0
    assert counts["unharmful"] > 0
    assert sum(counts.values()) == 3


@pytest.mark.network
def test_pilot_sample_matches_recorded_class_balance():
    """Pins the exact counts documented in the pilot config/report --
    genuinely depends on the real PolyGuardPrompts dataset's actual class
    distribution among the real seed=42/limit=50 draw, not just the
    sampling mechanism, so this can't be preserved with a synthetic
    fixture."""
    loader = PolyGuardPromptsLoader()
    items = loader.load("en", limit=PILOT_LIMIT, seed=PILOT_SEED)
    counts = Counter(i.expected_label for i in items)
    assert counts == {"harmful": 28, "unharmful": 22}


def test_pilot_sample_is_a_valid_subset_of_the_full_english_benchmark(tiny_polyguard_parquet):
    """Subset relationship is a property of the sampling mechanism, not of
    the real dataset's content -- safe on the tiny fixture."""
    loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
    full_ids = {i.parallel_item_id for i in loader.load("en")}
    pilot_ids = {i.parallel_item_id for i in loader.load("en", limit=2, seed=PILOT_SEED)}
    assert pilot_ids <= full_ids
    assert len(pilot_ids) == 2


@pytest.mark.network
def test_full_english_load_marks_exactly_the_documented_null_label_rows_as_none():
    """Cross-checks the loader's expected_label=None rows directly against
    the raw cached parquet (independent of pandas' None-vs-NaN
    representation for the missing cells -- see docs/phase1-notes.md's 26
    unresolved-label items), rather than trusting a hardcoded id list that
    could silently drift from the actual data. Genuinely requires the real
    full dataset: pins the real row count (1725) and the real null-label
    row count (26)."""
    loader = PolyGuardPromptsLoader()
    items = loader.load("en")
    assert len(items) == 1725

    df = pd.read_parquet(loader._ensure_cached())
    en = df[df["language"] == "English"]
    raw_null_ids = set(en.loc[en["prompt_harm_label"].apply(pd.isna), "id"].astype(str))

    none_label_ids = {i.parallel_item_id for i in items if i.expected_label is None}
    assert none_label_ids == raw_null_ids
    assert len(none_label_ids) == 26  # pinned per docs/phase1-notes.md

    assert all(i.expected_label != "nan" for i in items)
    assert all(i.expected_label in {"harmful", "unharmful", None} for i in items)


@pytest.mark.network
def test_pilot_sample_unaffected_by_null_label_handling_fix():
    """The 50-item pilot (seed=42) never draws one of the 26 null-label
    rows, so this fix must not introduce any None labels into it -- the
    recorded 28/22 class balance in test_pilot_sample_matches_recorded_class_balance
    already pins the rest. Genuinely requires the real dataset: this is a
    property of the real seed=42/limit=50 draw against the real null rows,
    not something a synthetic fixture can stand in for."""
    loader = PolyGuardPromptsLoader()
    items = loader.load("en", limit=PILOT_LIMIT, seed=PILOT_SEED)
    assert all(i.expected_label is not None for i in items)
