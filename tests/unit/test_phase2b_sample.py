"""Tests for the Phase 2B frozen 200-item parallel sample
(configs/samples/phase2b_polyguard_en_ko_200.json).

Split the same way as tests/unit/test_pilot_sampling.py: properties
checkable directly from the checked-in manifest file run network-free;
properties that require cross-referencing the real PolyGuardPrompts
dataset are marked @pytest.mark.network.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from mpe.datasets.base import deterministic_sample
from mpe.datasets.manifest import load_id_manifest
from mpe.datasets.polyguard import PolyGuardPromptsLoader

MANIFEST_PATH = Path("configs/samples/phase2b_polyguard_en_ko_200.json")


def _load_manifest_json() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


class TestManifestFileProperties:
    """Network-free: checkable directly from the checked-in file."""

    def test_manifest_has_exactly_200_ids(self):
        ids = load_id_manifest(MANIFEST_PATH)
        assert len(ids) == 200

    def test_manifest_reading_is_reproducible_across_calls(self):
        first = load_id_manifest(MANIFEST_PATH)
        second = load_id_manifest(MANIFEST_PATH)
        assert first == second

    def test_manifest_has_no_duplicate_ids(self):
        ids = load_id_manifest(MANIFEST_PATH)
        assert len(ids) == len(set(ids))

    def test_manifest_declares_both_languages(self):
        data = _load_manifest_json()
        assert set(data["languages"]) == {"en", "ko"}

    def test_manifest_labels_contain_no_null_or_unrecognized_values(self):
        data = _load_manifest_json()
        labels = data["labels"]
        assert set(labels.keys()) == set(data["parallel_item_ids"])
        assert all(v in {"harmful", "unharmful"} for v in labels.values())

    def test_manifest_declared_balance_matches_its_own_labels(self):
        data = _load_manifest_json()
        labels = list(data["labels"].values())
        assert labels.count("harmful") == data["harmful_count"]
        assert labels.count("unharmful") == data["unharmful_count"]
        assert data["harmful_count"] + data["unharmful_count"] == data["sample_size"] == 200

    def test_manifest_balance_is_not_forced_to_50_50(self):
        """Phase 2B explicitly preserves the natural class distribution --
        pins that this sample is NOT artificially balanced."""
        data = _load_manifest_json()
        assert data["harmful_count"] != data["unharmful_count"]
        assert data["harmful_count"] == 93
        assert data["unharmful_count"] == 107

    def test_manifest_is_byte_for_byte_unchanged_by_the_len1024_ablation(self):
        """The 1024-token truncation ablation (Phase 2B configs, len1024
        pilot configs) must change only max_new_tokens/name -- it must
        never touch the frozen sample itself. Hashing the raw file content
        catches any change (including reordering or whitespace), not just
        the aggregate properties already pinned above."""
        import hashlib

        digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        assert digest == "a09f239a3cee9f467bfdd5d7dc6ebbfd9b1ca521991e2cf47b458c8e550d210c"


class TestManifestVsRealDataset:
    """Requires the real cached PolyGuardPrompts parquet."""

    @pytest.mark.network
    def test_regenerating_from_real_data_reproduces_the_checked_in_manifest(self):
        """The manifest's own documented derivation (filter null-label ids,
        then deterministic_sample(seed=42) over the remainder) must
        reproduce this exact checked-in file -- pins true reproducibility,
        not just "the file doesn't change when read twice"."""
        loader = PolyGuardPromptsLoader()
        df = pd.read_parquet(loader._ensure_cached())
        en = df[df["language"] == "English"]
        null_ids = set(en.loc[en["prompt_harm_label"].apply(pd.isna), "id"].astype(str))
        non_null = en[~en["id"].astype(str).isin(null_ids)]

        data = _load_manifest_json()
        regenerated = deterministic_sample(non_null["id"].tolist(), 200, seed=data["seed"])
        regenerated_str = sorted(str(x) for x in regenerated)

        assert regenerated_str == sorted(load_id_manifest(MANIFEST_PATH))

    @pytest.mark.network
    def test_manifest_labels_match_the_real_dataset(self):
        data = _load_manifest_json()
        loader = PolyGuardPromptsLoader()
        df = pd.read_parquet(loader._ensure_cached())
        en = df[df["language"] == "English"]
        real_labels = dict(zip(en["id"].astype(str), en["prompt_harm_label"]))
        for item_id, label in data["labels"].items():
            assert real_labels[item_id] == label

    @pytest.mark.network
    def test_manifest_ids_have_zero_overlap_with_the_real_null_label_ids(self):
        loader = PolyGuardPromptsLoader()
        df = pd.read_parquet(loader._ensure_cached())
        en = df[df["language"] == "English"]
        real_null_ids = set(en.loc[en["prompt_harm_label"].apply(pd.isna), "id"].astype(str))
        manifest_ids = set(load_id_manifest(MANIFEST_PATH))
        assert manifest_ids.isdisjoint(real_null_ids)

    @pytest.mark.network
    def test_en_and_ko_resolve_to_the_same_200_ids_via_the_manifest(self):
        """The core cross-lingual guarantee: loading the manifest's ids for
        EN and for KO must yield the identical parallel_item_id set,
        confirming EN and KO refer to the same underlying items."""
        loader = PolyGuardPromptsLoader()
        ids = load_id_manifest(MANIFEST_PATH)
        en_items = loader.load("en", item_ids=ids)
        ko_items = loader.load("ko", item_ids=ids)
        assert len(en_items) == len(ko_items) == 200
        assert {i.parallel_item_id for i in en_items} == {i.parallel_item_id for i in ko_items} == set(ids)

    @pytest.mark.network
    def test_en_and_ko_expected_labels_agree_for_every_item(self):
        loader = PolyGuardPromptsLoader()
        ids = load_id_manifest(MANIFEST_PATH)
        en_by_id = {i.parallel_item_id: i.expected_label for i in loader.load("en", item_ids=ids)}
        ko_by_id = {i.parallel_item_id: i.expected_label for i in loader.load("ko", item_ids=ids)}
        assert en_by_id == ko_by_id
