import pytest

from mpe.datasets import BelebeleKoLoader, PolyGuardPromptsLoader, TaskType, XSTestLoader


class TestPolyGuardPromptsLoader:
    def test_loads_and_filters_by_language(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        en_items = loader.load("en")
        ko_items = loader.load("ko")
        assert len(en_items) == 3
        assert len(ko_items) == 3
        assert all(i.language == "en" for i in en_items)
        assert all(i.language == "ko" for i in ko_items)

    def test_en_ko_calls_with_same_seed_are_parallel(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        en_items = loader.load("en", limit=2, seed=42)
        ko_items = loader.load("ko", limit=2, seed=42)
        assert {i.parallel_item_id for i in en_items} == {i.parallel_item_id for i in ko_items}

    def test_task_type_and_expected_label(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        items = loader.load("en")
        assert all(i.task_type == TaskType.REFUSAL_CLASSIFICATION for i in items)
        harmful = [i for i in items if i.parallel_item_id == "0"][0]
        assert harmful.expected_label == "harmful"

    def test_unsupported_language_raises(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        with pytest.raises(ValueError, match="Unsupported language"):
            loader.load("fr")

    def test_limit_exceeding_available_raises(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        with pytest.raises(ValueError, match="exceeds"):
            loader.load("en", limit=100)

    def test_missing_label_rows_become_expected_label_none_regardless_of_nan_or_none(
        self, tiny_polyguard_parquet_with_null_labels
    ):
        """Pins the fix for the pandas-2-vs-3 null-representation drift bug:
        a NaN-labeled row and a None-labeled row must both resolve to
        expected_label=None, never crash, and never become the literal
        string "nan"."""
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet_with_null_labels)
        items = loader.load("en")
        assert len(items) == 4

        by_id = {i.parallel_item_id: i for i in items}
        assert by_id["0"].expected_label == "harmful"
        assert by_id["1"].expected_label == "unharmful"
        assert by_id["2"].expected_label is None  # was float('nan')
        assert by_id["3"].expected_label is None  # was None

        assert all(i.expected_label != "nan" for i in items)

    def test_null_label_rows_still_pair_correctly_across_en_and_ko(
        self, tiny_polyguard_parquet_with_null_labels
    ):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet_with_null_labels)
        en_items = loader.load("en")
        ko_items = loader.load("ko")
        assert {i.parallel_item_id for i in en_items} == {i.parallel_item_id for i in ko_items}
        ko_by_id = {i.parallel_item_id: i for i in ko_items}
        assert ko_by_id["2"].expected_label is None
        assert ko_by_id["3"].expected_label is None


class TestPolyGuardPromptsLoaderItemIds:
    """The frozen-sample path (Phase 2B): explicit item_ids instead of
    limit/seed-driven deterministic_sample()."""

    def test_item_ids_returns_exactly_those_items(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        items = loader.load("en", item_ids=["0", "2"])
        assert {i.parallel_item_id for i in items} == {"0", "2"}
        assert len(items) == 2

    def test_item_ids_ignores_limit_and_seed(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        # limit=1 would normally raise nothing (it's just a smaller sample),
        # but must be irrelevant entirely once item_ids is given: all 2
        # requested ids come back regardless of limit/seed.
        items = loader.load("en", limit=1, seed=999, item_ids=["0", "1"])
        assert {i.parallel_item_id for i in items} == {"0", "1"}

    def test_missing_item_id_raises_clearly(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        with pytest.raises(ValueError, match="not present"):
            loader.load("en", item_ids=["0", "does-not-exist"])

    def test_item_ids_still_pair_correctly_across_en_and_ko(self, tiny_polyguard_parquet):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet)
        en_items = loader.load("en", item_ids=["0", "1"])
        ko_items = loader.load("ko", item_ids=["0", "1"])
        assert {i.parallel_item_id for i in en_items} == {i.parallel_item_id for i in ko_items} == {"0", "1"}

    def test_item_ids_with_null_label_row_resolves_expected_label_none(
        self, tiny_polyguard_parquet_with_null_labels
    ):
        loader = PolyGuardPromptsLoader(cache_path=tiny_polyguard_parquet_with_null_labels)
        items = loader.load("en", item_ids=["2"])
        assert len(items) == 1
        assert items[0].expected_label is None


class TestXSTestLoader:
    """Uses the real committed local snapshots -- no network, no fixtures."""

    def test_english_loads_full_450(self):
        loader = XSTestLoader()
        items = loader.load("en")
        assert len(items) == 450
        assert all(i.task_type == TaskType.REFUSAL_CLASSIFICATION for i in items)
        assert {i.expected_label for i in items} <= {"safe", "unsafe"}

    def test_korean_draft_loads_16_items(self):
        loader = XSTestLoader()
        items = loader.load("ko")
        assert len(items) == 16
        assert all(i.language == "ko" for i in items)
        assert all(i.metadata["translation_status"] == "draft_ai_translated_pending_native_review" for i in items)

    def test_korean_limit_beyond_draft_size_raises(self):
        loader = XSTestLoader()
        with pytest.raises(ValueError, match="exceeds"):
            loader.load("ko", limit=50)

    def test_en_ko_share_parallel_ids_for_the_translated_subset(self):
        loader = XSTestLoader()
        en_ids = {i.parallel_item_id for i in loader.load("en")}
        ko_ids = {i.parallel_item_id for i in loader.load("ko")}
        assert ko_ids <= en_ids


class TestBelebeleKoLoader:
    def test_loads_and_pairs_via_link_and_question_number(self, tiny_belebele_cache):
        loader = BelebeleKoLoader(cache_dir=tiny_belebele_cache)
        en_items = loader.load("en")
        ko_items = loader.load("ko")
        assert len(en_items) == 3
        assert {i.parallel_item_id for i in en_items} == {i.parallel_item_id for i in ko_items}

    def test_multiple_choice_fields_populated(self, tiny_belebele_cache):
        loader = BelebeleKoLoader(cache_dir=tiny_belebele_cache)
        items = loader.load("en")
        item = items[0]
        assert item.task_type == TaskType.MULTIPLE_CHOICE
        assert item.choices == ["A", "B", "C", "D"]
        assert item.correct_choice_index == 1  # correct_answer_num "2" -> index 1
