from mpe.datasets.schema import BenchmarkItem, TaskType
from mpe.evaluators.schema import RawResponse
from mpe.parsing.output_parser import classify_refusal, detect_language, parse_mc_choice, parse_output


def _refusal_item(language: str, expected_label: str) -> BenchmarkItem:
    return BenchmarkItem(
        item_id="x",
        benchmark="test",
        language=language,
        parallel_item_id="1",
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="prompt",
        expected_label=expected_label,
    )


def _mc_item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="mc1",
        benchmark="test",
        language="en",
        parallel_item_id="1",
        task_type=TaskType.MULTIPLE_CHOICE,
        prompt="prompt",
        choices=["a", "b", "c", "d"],
        correct_choice_index=2,
    )


class TestDetectLanguage:
    def test_english_text(self):
        assert detect_language("I'm sorry, I can't help with that.") == "en"

    def test_korean_text(self):
        assert detect_language("죄송하지만 도와드릴 수 없습니다.") == "ko"

    def test_empty_text_returns_none(self):
        assert detect_language("") is None

    def test_mixed_majority_korean(self):
        # 6 Hangul chars (정답은입니다) vs 1 Latin char (A) -- Korean should win.
        assert detect_language("정답은 A 입니다") == "ko"


class TestClassifyRefusal:
    def test_english_refusal_phrases(self):
        assert classify_refusal("I'm sorry, but I can't help with that.", "en") == "refusal"
        assert classify_refusal("I cannot provide that information.", "en") == "refusal"

    def test_english_compliance(self):
        assert classify_refusal("Sure, here's how to do it.", "en") == "compliance"

    def test_korean_refusal_phrases(self):
        assert classify_refusal("죄송하지만 도와드릴 수 없습니다.", "ko") == "refusal"

    def test_korean_compliance(self):
        assert classify_refusal("네, 알려드리겠습니다.", "ko") == "compliance"


class TestParseMcChoice:
    def test_answer_prefix(self):
        assert parse_mc_choice("Answer: 3") == 2

    def test_bare_digit(self):
        assert parse_mc_choice("2") == 1

    def test_korean_prefix(self):
        assert parse_mc_choice("정답: 4") == 3

    def test_no_valid_choice(self):
        assert parse_mc_choice("I'm not sure.") is None


class TestParseOutput:
    def test_empty_completion_is_not_parseable(self):
        item = _refusal_item("en", "harmful")
        response = RawResponse(item_id="x", completion="")
        parsed = parse_output(item, response)
        assert parsed.is_parseable is False
        assert parsed.refusal_label is None

    def test_refusal_classification_end_to_end(self):
        item = _refusal_item("en", "harmful")
        response = RawResponse(item_id="x", completion="I'm sorry, I can't help with that.")
        parsed = parse_output(item, response)
        assert parsed.is_parseable is True
        assert parsed.refusal_label == "refusal"
        assert parsed.language_match is True

    def test_language_mismatch_detected(self):
        item = _refusal_item("ko", "harmful")
        response = RawResponse(item_id="x", completion="I'm sorry, I can't help with that.")
        parsed = parse_output(item, response)
        assert parsed.language_match is False

    def test_mc_valid_choice(self):
        item = _mc_item()
        response = RawResponse(item_id="mc1", completion="Answer: 3")
        parsed = parse_output(item, response)
        assert parsed.is_parseable is True
        assert parsed.mc_choice_index == 2

    def test_mc_unparseable_choice_marks_not_parseable(self):
        item = _mc_item()
        response = RawResponse(item_id="mc1", completion="I don't know.")
        parsed = parse_output(item, response)
        assert parsed.is_parseable is False
        assert parsed.mc_choice_index is None
