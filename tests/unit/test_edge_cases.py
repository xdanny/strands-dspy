"""Edge case and error handling tests for strands_dspy."""

import dspy

from strands_dspy.helpers import (
    contains_match,
    exact_match,
    extract_first_user_text,
    extract_last_assistant_text,
)


class TestExtractorEdgeCases:
    """Edge cases for extractors."""

    def test_extractor_with_none_messages(self):
        """Test extractor handles None input gracefully."""
        # Should not crash, but will likely return empty
        try:
            result = extract_first_user_text(None)
            # If it doesn't crash, check it returns something sensible
            assert isinstance(result, dict)
        except (TypeError, AttributeError):
            # It's OK if it raises an error for None input
            pass

    def test_extractor_with_malformed_message_structure(self):
        """Test extractor with unexpected message structure."""
        messages = [
            {
                "role": "user",
                "content": "not a list",  # Should be a list
            }
        ]
        result = extract_first_user_text(messages)
        assert result == {"question": ""}

    def test_extractor_with_missing_text_key(self):
        """Test extractor when content block missing text key."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text"}],  # No "text" key
            }
        ]
        result = extract_first_user_text(messages)
        assert result == {"question": ""}

    def test_extractor_with_mixed_content_types(self):
        """Test extractor with various content block types."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "http://example.com"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        result = extract_first_user_text(messages)
        assert result == {"question": "What is this?"}

    def test_output_extractor_with_list_content(self):
        """Test output extractor when content is unexpectedly a list of strings."""

        class MockResult:
            def __init__(self):
                self.message = {"content": ["string1", "string2"]}

        result_obj = MockResult()
        result = extract_last_assistant_text(result_obj)
        # Should handle gracefully - likely returns empty
        assert result == {"answer": ""}


class TestMetricEdgeCases:
    """Edge cases for metrics."""

    def test_metric_with_none_values(self):
        """Test metric when values are None."""
        example = dspy.Example(answer=None)
        prediction = dspy.Prediction(answer=None)

        # Should not crash
        score = exact_match(example, prediction)
        # "None" == "None" after str() conversion
        assert score == 1.0

    def test_metric_with_numeric_answers(self):
        """Test metric with numeric (non-string) answers."""
        example = dspy.Example(answer=42)
        prediction = dspy.Prediction(answer=42)

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_metric_with_empty_strings(self):
        """Test metric with empty string answers."""
        example = dspy.Example(answer="")
        prediction = dspy.Prediction(answer="")

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_metric_with_very_long_strings(self):
        """Test metric with very long strings."""
        long_string = "a" * 10000
        example = dspy.Example(answer=long_string)
        prediction = dspy.Prediction(answer=long_string)

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_metric_with_unicode_characters(self):
        """Test metric with unicode characters."""
        example = dspy.Example(answer="こんにちは")
        prediction = dspy.Prediction(answer="こんにちは")

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_metric_with_special_characters(self):
        """Test metric with special characters."""
        example = dspy.Example(answer="Hello\nWorld\t!")
        prediction = dspy.Prediction(answer="Hello\nWorld\t!")

        score = exact_match(example, prediction)
        assert score == 1.0

    def test_contains_match_with_empty_substring(self):
        """Test contains match when expected answer is empty."""
        example = dspy.Example(answer="")
        prediction = dspy.Prediction(answer="Some text")

        score = contains_match(example, prediction)
        # Empty string is contained in everything
        assert score == 1.0

    def test_metric_with_list_or_dict_answer(self):
        """Test metric when answer is not a string."""
        example = dspy.Example(answer=["item1", "item2"])
        prediction = dspy.Prediction(answer=["item1", "item2"])

        # Should convert to string and compare
        score = exact_match(example, prediction)
        # str(list) should match
        assert score == 1.0


class TestSuccessCriteriaEdgeCases:
    """Edge cases for success criteria."""

    def test_success_criteria_with_none_input(self):
        """Test success criteria with None input."""
        from strands_dspy.helpers import always_success

        # Should not crash
        is_success, score = always_success(None)
        assert is_success is True
        assert score == 1.0

    def test_min_length_with_very_large_threshold(self):
        """Test min_length_success with unrealistic threshold."""
        from strands_dspy.helpers import min_length_success

        class MockResult:
            def __init__(self):
                self.message = {"content": [{"type": "text", "text": "Short text"}]}

        result = MockResult()
        check_length = min_length_success(min_chars=1000000)
        is_success, score = check_length(result)

        assert is_success is False
        assert score == 0.0

    def test_min_length_with_zero_threshold(self):
        """Test min_length_success with zero threshold."""
        from strands_dspy.helpers import min_length_success

        class MockResult:
            def __init__(self):
                self.message = {"content": [{"type": "text", "text": ""}]}

        result = MockResult()
        check_length = min_length_success(min_chars=0)
        is_success, score = check_length(result)

        # Even empty string meets 0 threshold
        assert is_success is True
        assert score == 1.0

    def test_combine_criteria_with_no_criteria(self):
        """Test combine_criteria with empty list."""
        from strands_dspy.helpers import combine_criteria

        class MockResult:
            pass

        result = MockResult()

        # Edge case: what happens with no criteria?
        # Should probably return success with score 1.0
        combined = combine_criteria()
        is_success, score = combined(result)

        # With no criteria, should succeed
        assert is_success is True
        assert score == 1.0

    def test_combine_criteria_with_single_criterion(self):
        """Test combine_criteria with single criterion."""
        from strands_dspy.helpers import always_success, combine_criteria

        class MockResult:
            pass

        result = MockResult()
        combined = combine_criteria(always_success)
        is_success, score = combined(result)

        assert is_success is True
        assert score == 1.0


class TestConcurrencyAndThreadSafety:
    """Tests for concurrent usage (basic thread safety)."""

    def test_extractors_with_concurrent_access(self):
        """Test that extractors can be used concurrently."""
        import threading

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Test question"}],
            }
        ]

        results = []

        def extract_multiple_times():
            for _ in range(100):
                result = extract_first_user_text(messages)
                results.append(result)

        threads = [threading.Thread(target=extract_multiple_times) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be identical
        assert all(r == {"question": "Test question"} for r in results)
        assert len(results) == 1000  # 10 threads * 100 iterations


class TestPerformance:
    """Basic performance tests."""

    def test_metric_performance_with_large_dataset(self):
        """Test metric performance doesn't degrade significantly."""
        import time

        # Create 1000 example/prediction pairs
        examples = [dspy.Example(answer=f"answer_{i}") for i in range(1000)]
        predictions = [dspy.Prediction(answer=f"answer_{i}") for i in range(1000)]

        start = time.time()

        for example, prediction in zip(examples, predictions, strict=True):
            exact_match(example, prediction)

        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second for 1000 comparisons)
        assert elapsed < 1.0, f"Metric took {elapsed}s for 1000 comparisons"
