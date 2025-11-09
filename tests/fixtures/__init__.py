"""Test fixtures for strands-dspy tests."""

from .golden_dataset import (
    GOLDEN_QA_PAIRS,
    get_golden_dataset,
    split_golden_dataset,
    exact_match_metric,
    contains_metric,
    gepa_feedback_metric,
)

from .hard_dataset import (
    HARD_REASONING_PAIRS,
    get_hard_dataset,
    split_hard_dataset,
    hard_reasoning_metric,
    simple_contains_metric,
)

__all__ = [
    "GOLDEN_QA_PAIRS",
    "get_golden_dataset",
    "split_golden_dataset",
    "exact_match_metric",
    "contains_metric",
    "gepa_feedback_metric",
    "HARD_REASONING_PAIRS",
    "get_hard_dataset",
    "split_hard_dataset",
    "hard_reasoning_metric",
    "simple_contains_metric",
]
