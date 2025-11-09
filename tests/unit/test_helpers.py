"""Comprehensive tests for strands_dspy.helpers module."""

import dspy

from strands_dspy.helpers import (
    # Success criteria
    always_success,
    combine_criteria,
    combine_extractors,
    contains_match,
    contains_match_gepa,
    end_turn_success,
    # Metrics
    exact_match,
    exact_match_gepa,
    extract_all_user_messages,
    extract_field_from_message,
    # Extractors
    extract_first_user_text,
    extract_last_assistant_text,
    make_gepa_metric,
    min_length_success,
    no_error_success,
    numeric_match,
    numeric_match_gepa,
)

# =============================================================================
# EXTRACTOR TESTS
# =============================================================================


class TestExtractors:
    """Tests for input/output extractors."""

    def test_extract_first_user_text_normal(self):
        """Test extracting first user message text."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is Python?"}],
            }
        ]
        result = extract_first_user_text(messages)
        assert result == {"question": "What is Python?"}

    def test_extract_first_user_text_empty_messages(self):
        """Test with empty messages list."""
        messages = []
        result = extract_first_user_text(messages)
        assert result == {"question": ""}

    def test_extract_first_user_text_no_user_messages(self):
        """Test when there are no user messages."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]
        result = extract_first_user_text(messages)
        assert result == {"question": ""}

    def test_extract_first_user_text_skips_non_user(self):
        """Test that it skips assistant messages and gets first user."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Assistant msg"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "User msg"}],
            },
        ]
        result = extract_first_user_text(messages)
        assert result == {"question": "User msg"}

    def test_extract_last_assistant_text_normal(self):
        """Test extracting last assistant message text."""

        class MockResult:
            def __init__(self):
                self.message = {
                    "content": [{"type": "text", "text": "Python is a programming language"}]
                }

        result_obj = MockResult()
        result = extract_last_assistant_text(result_obj)
        assert result == {"answer": "Python is a programming language"}

    def test_extract_last_assistant_text_no_message(self):
        """Test when result has no message attribute."""

        class MockResult:
            pass

        result_obj = MockResult()
        result = extract_last_assistant_text(result_obj)
        assert result == {"answer": ""}

    def test_extract_last_assistant_text_empty_content(self):
        """Test when message content is empty."""

        class MockResult:
            def __init__(self):
                self.message = {"content": []}

        result_obj = MockResult()
        result = extract_last_assistant_text(result_obj)
        assert result == {"answer": ""}

    def test_extract_all_user_messages_multiple(self):
        """Test extracting all user messages."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "First question"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "First answer"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Second question"}],
            },
        ]
        result = extract_all_user_messages(messages)
        assert result == {"questions": ["First question", "Second question"]}

    def test_extract_all_user_messages_none(self):
        """Test when there are no user messages."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Assistant only"}],
            }
        ]
        result = extract_all_user_messages(messages)
        assert result == {"questions": []}

    def test_extract_field_from_message_default(self):
        """Test extracting custom field with defaults."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Question text"}],
            }
        ]
        result = extract_field_from_message(messages)
        assert result == {"question": "Question text"}

    def test_extract_field_from_message_custom_field(self):
        """Test extracting with custom field name."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "System prompt"}],
            }
        ]
        result = extract_field_from_message(messages, field_name="context", role="system")
        assert result == {"context": "System prompt"}

    def test_extract_field_from_message_with_index(self):
        """Test extracting second message of a role."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "First"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Second"}],
            },
        ]
        result = extract_field_from_message(messages, field_name="followup", role="user", index=1)
        assert result == {"followup": "Second"}

    def test_extract_field_from_message_index_out_of_range(self):
        """Test when index is out of range."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Only one"}],
            }
        ]
        result = extract_field_from_message(messages, field_name="missing", role="user", index=5)
        assert result == {"missing": ""}

    def test_combine_extractors(self):
        """Test combining multiple extractors."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "User question"}],
            }
        ]

        def custom_extractor(messages):
            return {"custom_field": "custom_value"}

        combined = combine_extractors(extract_first_user_text, custom_extractor)
        result = combined(messages)

        assert result == {"question": "User question", "custom_field": "custom_value"}


# =============================================================================
# METRIC TESTS
# =============================================================================


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_exact_match_success(self):
        """Test exact match when strings match."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="Paris")

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_exact_match_case_insensitive(self):
        """Test that exact match is case insensitive."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="PARIS")

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_exact_match_strips_whitespace(self):
        """Test that exact match strips whitespace."""
        example = dspy.Example(answer="  Paris  ")
        prediction = dspy.Prediction(answer="Paris")

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_exact_match_failure(self):
        """Test exact match when strings don't match."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="London")

        score = exact_match(example, prediction)
        assert score == 0.0

    def test_exact_match_missing_attribute(self):
        """Test exact match when prediction is missing answer attribute."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction()  # No answer attribute

        score = exact_match(example, prediction)
        assert score == 0.0

    def test_contains_match_success(self):
        """Test contains match when answer is substring."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="The capital is Paris, France")

        score = contains_match(example, prediction)
        assert score == 1.0

    def test_contains_match_case_insensitive(self):
        """Test contains match is case insensitive."""
        example = dspy.Example(answer="paris")
        prediction = dspy.Prediction(answer="The capital is PARIS")

        score = contains_match(example, prediction)
        assert score == 1.0

    def test_contains_match_failure(self):
        """Test contains match when substring not found."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="London is the capital")

        score = contains_match(example, prediction)
        assert score == 0.0

    def test_numeric_match_exact(self):
        """Test numeric match with exact number."""
        example = dspy.Example(answer="42")
        prediction = dspy.Prediction(answer="The answer is 42")

        score = numeric_match(example, prediction)
        assert score == 1.0

    def test_numeric_match_within_tolerance(self):
        """Test numeric match within 1% tolerance."""
        example = dspy.Example(answer="100")
        prediction = dspy.Prediction(answer="100.5")  # Within 1%

        score = numeric_match(example, prediction)
        assert score == 0.9

    def test_numeric_match_outside_tolerance(self):
        """Test numeric match outside tolerance."""
        example = dspy.Example(answer="100")
        prediction = dspy.Prediction(answer="200")

        score = numeric_match(example, prediction)
        assert score == 0.0

    def test_numeric_match_no_numbers_fallback(self):
        """Test numeric match falls back to contains when no numbers."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="The answer is Paris")

        score = numeric_match(example, prediction)
        assert score == 1.0  # Falls back to contains match

    def test_numeric_match_negative_numbers(self):
        """Test numeric match with negative numbers."""
        example = dspy.Example(answer="-42")
        prediction = dspy.Prediction(answer="The answer is -42")

        score = numeric_match(example, prediction)
        assert score == 1.0

    def test_numeric_match_decimal_numbers(self):
        """Test numeric match with decimal numbers."""
        example = dspy.Example(answer="3.14")
        prediction = dspy.Prediction(answer="Pi is approximately 3.14")

        score = numeric_match(example, prediction)
        assert score == 1.0

    def test_make_gepa_metric_basic(self):
        """Test converting simple metric to GEPA format."""

        def simple_metric(example, prediction, trace=None):
            return 1.0 if example.answer == prediction.answer else 0.0

        gepa_metric = make_gepa_metric(simple_metric)

        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="Paris")

        result = gepa_metric(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "score")
        assert hasattr(result, "feedback")
        assert result.score == 1.0
        assert "Perfect" in result.feedback or "correct" in result.feedback.lower()

    def test_make_gepa_metric_with_custom_feedback(self):
        """Test GEPA metric with custom feedback function."""

        def simple_metric(example, prediction, trace=None):
            return 0.5

        def custom_feedback(example, prediction, score):
            return f"Custom feedback: score={score}"

        gepa_metric = make_gepa_metric(simple_metric, custom_feedback)

        example = dspy.Example(answer="test")
        prediction = dspy.Prediction(answer="test")

        result = gepa_metric(example, prediction)

        assert result.score == 0.5
        assert result.feedback == "Custom feedback: score=0.5"

    def test_exact_match_gepa(self):
        """Test pre-made exact_match_gepa metric."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="Paris")

        result = exact_match_gepa(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert result.score == 1.0

    def test_contains_match_gepa(self):
        """Test pre-made contains_match_gepa metric."""
        example = dspy.Example(answer="Paris")
        prediction = dspy.Prediction(answer="The capital is Paris")

        result = contains_match_gepa(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert result.score == 1.0

    def test_numeric_match_gepa(self):
        """Test pre-made numeric_match_gepa metric."""
        example = dspy.Example(answer="42")
        prediction = dspy.Prediction(answer="42")

        result = numeric_match_gepa(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert result.score == 1.0


# =============================================================================
# SUCCESS CRITERIA TESTS
# =============================================================================


class TestSuccessCriteria:
    """Tests for success criteria functions."""

    def test_always_success(self):
        """Test always_success returns True for any input."""

        class MockResult:
            pass

        result = MockResult()
        is_success, score = always_success(result)

        assert is_success is True
        assert score == 1.0

    def test_end_turn_success_true(self):
        """Test end_turn_success when stop_reason is end_turn."""

        class MockResult:
            def __init__(self):
                self.stop_reason = "end_turn"

        result = MockResult()
        is_success, score = end_turn_success(result)

        assert is_success is True
        assert score == 1.0

    def test_end_turn_success_max_tokens(self):
        """Test end_turn_success fails for max_tokens."""

        class MockResult:
            def __init__(self):
                self.stop_reason = "max_tokens"

        result = MockResult()
        is_success, score = end_turn_success(result)

        assert is_success is False
        assert score == 0.0

    def test_end_turn_success_no_attribute(self):
        """Test end_turn_success when result has no stop_reason."""

        class MockResult:
            pass

        result = MockResult()
        is_success, score = end_turn_success(result)

        assert is_success is False
        assert score == 0.0

    def test_no_error_success_clean(self):
        """Test no_error_success with clean result."""

        class MockResult:
            pass

        result = MockResult()
        is_success, score = no_error_success(result)

        assert is_success is True
        assert score == 1.0

    def test_no_error_success_with_error_attribute(self):
        """Test no_error_success when error attribute exists."""

        class MockResult:
            def __init__(self):
                self.error = "Some error occurred"

        result = MockResult()
        is_success, score = no_error_success(result)

        assert is_success is False
        assert score == 0.0

    def test_no_error_success_with_exception(self):
        """Test no_error_success when exception attribute exists."""

        class MockResult:
            def __init__(self):
                self.exception = Exception("Test exception")

        result = MockResult()
        is_success, score = no_error_success(result)

        assert is_success is False
        assert score == 0.0

    def test_no_error_success_stop_reason_error(self):
        """Test no_error_success when stop_reason is error."""

        class MockResult:
            def __init__(self):
                self.stop_reason = "error"

        result = MockResult()
        is_success, score = no_error_success(result)

        assert is_success is False
        assert score == 0.0

    def test_min_length_success_meets_threshold(self):
        """Test min_length_success when length meets threshold."""

        class MockResult:
            def __init__(self):
                self.message = {
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a long enough answer that meets the threshold",
                        }
                    ]
                }

        result = MockResult()
        check_length = min_length_success(min_chars=20)
        is_success, score = check_length(result)

        assert is_success is True
        assert score == 1.0

    def test_min_length_success_below_threshold(self):
        """Test min_length_success when length is below threshold."""

        class MockResult:
            def __init__(self):
                self.message = {"content": [{"type": "text", "text": "Short"}]}

        result = MockResult()
        check_length = min_length_success(min_chars=50)
        is_success, score = check_length(result)

        assert is_success is False
        assert score == 0.0

    def test_min_length_success_no_message(self):
        """Test min_length_success when result has no message."""

        class MockResult:
            pass

        result = MockResult()
        check_length = min_length_success(min_chars=10)
        is_success, score = check_length(result)

        assert is_success is False
        assert score == 0.0

    def test_min_length_success_empty_content(self):
        """Test min_length_success with empty content."""

        class MockResult:
            def __init__(self):
                self.message = {"content": []}

        result = MockResult()
        check_length = min_length_success(min_chars=10)
        is_success, score = check_length(result)

        assert is_success is False
        assert score == 0.0

    def test_combine_criteria_all_pass(self):
        """Test combine_criteria when all criteria pass."""

        class MockResult:
            def __init__(self):
                self.stop_reason = "end_turn"

        result = MockResult()
        combined = combine_criteria(always_success, end_turn_success, no_error_success)
        is_success, score = combined(result)

        assert is_success is True
        assert score == 1.0

    def test_combine_criteria_one_fails(self):
        """Test combine_criteria when one criterion fails."""

        class MockResult:
            def __init__(self):
                self.stop_reason = "max_tokens"  # This will fail end_turn_success

        result = MockResult()
        combined = combine_criteria(always_success, end_turn_success)
        is_success, score = combined(result)

        assert is_success is False
        assert score == 0.0

    def test_combine_criteria_returns_min_score(self):
        """Test combine_criteria returns minimum score when all pass."""

        def partial_success(result):
            return True, 0.8

        def full_success(result):
            return True, 1.0

        class MockResult:
            pass

        result = MockResult()
        combined = combine_criteria(partial_success, full_success)
        is_success, score = combined(result)

        assert is_success is True
        assert score == 0.8  # Min of 0.8 and 1.0
