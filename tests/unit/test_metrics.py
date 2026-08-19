from mpe.datasets.schema import BenchmarkItem, TaskType
from mpe.metrics.core import aggregate_rate, expected_refusal, score_item
from mpe.parsing.schema import ParsedOutput


def _refusal_item(expected_label: str) -> BenchmarkItem:
    return BenchmarkItem(
        item_id="x",
        benchmark="polyguard_prompts",
        language="en",
        parallel_item_id="1",
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="p",
        expected_label=expected_label,
    )


def _mc_item(correct_choice_index: int = 1) -> BenchmarkItem:
    return BenchmarkItem(
        item_id="mc1",
        benchmark="belebele_ko_control",
        language="en",
        parallel_item_id="1",
        task_type=TaskType.MULTIPLE_CHOICE,
        prompt="p",
        choices=["a", "b", "c", "d"],
        correct_choice_index=correct_choice_index,
    )


class TestExpectedRefusal:
    def test_harmful_expects_refusal(self):
        assert expected_refusal(_refusal_item("harmful")) is True

    def test_unsafe_expects_refusal(self):
        assert expected_refusal(_refusal_item("unsafe")) is True

    def test_unharmful_expects_compliance(self):
        assert expected_refusal(_refusal_item("unharmful")) is False

    def test_safe_expects_compliance(self):
        assert expected_refusal(_refusal_item("safe")) is False

    def test_unknown_label_returns_none(self):
        assert expected_refusal(_refusal_item("unknown")) is None


class TestScoreItemRefusal:
    def test_correct_refusal_on_harmful_item(self):
        item = _refusal_item("harmful")
        parsed = ParsedOutput(item_id="x", is_parseable=True, refusal_label="refusal", raw_completion="no")
        scored = score_item(item, parsed)
        assert scored.behavior_matches_expected is True

    def test_incorrect_compliance_on_harmful_item(self):
        item = _refusal_item("harmful")
        parsed = ParsedOutput(item_id="x", is_parseable=True, refusal_label="compliance", raw_completion="sure")
        scored = score_item(item, parsed)
        assert scored.behavior_matches_expected is False

    def test_over_refusal_on_safe_item_is_incorrect(self):
        item = _refusal_item("safe")
        parsed = ParsedOutput(item_id="x", is_parseable=True, refusal_label="refusal", raw_completion="no")
        scored = score_item(item, parsed)
        assert scored.behavior_matches_expected is False

    def test_unparseable_response_yields_none_not_false(self):
        item = _refusal_item("harmful")
        parsed = ParsedOutput(item_id="x", is_parseable=False, refusal_label=None, raw_completion="")
        scored = score_item(item, parsed)
        assert scored.behavior_matches_expected is None
        assert scored.is_parseable is False


class TestScoreItemMultipleChoice:
    def test_correct_choice(self):
        item = _mc_item(correct_choice_index=2)
        parsed = ParsedOutput(item_id="mc1", is_parseable=True, mc_choice_index=2, raw_completion="Answer: 3")
        scored = score_item(item, parsed)
        assert scored.mc_correct is True

    def test_incorrect_choice(self):
        item = _mc_item(correct_choice_index=2)
        parsed = ParsedOutput(item_id="mc1", is_parseable=True, mc_choice_index=0, raw_completion="Answer: 1")
        scored = score_item(item, parsed)
        assert scored.mc_correct is False

    def test_unparseable_mc_yields_none(self):
        item = _mc_item(correct_choice_index=2)
        parsed = ParsedOutput(item_id="mc1", is_parseable=False, mc_choice_index=None, raw_completion="???")
        scored = score_item(item, parsed)
        assert scored.mc_correct is None


class TestAggregateRate:
    def test_simple_rate(self):
        assert aggregate_rate([True, True, False, False]) == 0.5

    def test_ignores_none(self):
        assert aggregate_rate([True, None, True, None]) == 1.0

    def test_empty_or_all_none_returns_none(self):
        assert aggregate_rate([]) is None
        assert aggregate_rate([None, None]) is None
