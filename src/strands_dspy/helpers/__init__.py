"""Helper functions for common strands-dspy patterns.

This module provides pre-built extractors, metrics, and success criteria
to make it easier to get started with DSPy optimization.
"""

from .extractors import (
    extract_first_user_text,
    extract_last_assistant_text,
    extract_all_user_messages,
    extract_field_from_message,
    combine_extractors,
)

from .metrics import (
    exact_match,
    contains_match,
    numeric_match,
    make_gepa_metric,
    exact_match_gepa,
    contains_match_gepa,
    numeric_match_gepa,
)

from .success import (
    always_success,
    end_turn_success,
    no_error_success,
    min_length_success,
    combine_criteria,
)

__all__ = [
    # Extractors
    "extract_first_user_text",
    "extract_last_assistant_text",
    "extract_all_user_messages",
    "extract_field_from_message",
    "combine_extractors",
    # Metrics
    "exact_match",
    "contains_match",
    "numeric_match",
    "make_gepa_metric",
    "exact_match_gepa",
    "contains_match_gepa",
    "numeric_match_gepa",
    # Success criteria
    "always_success",
    "end_turn_success",
    "no_error_success",
    "min_length_success",
    "combine_criteria",
]
