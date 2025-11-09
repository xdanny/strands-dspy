"""Pre-built metrics for DSPy optimization."""

import re
from collections.abc import Callable

import dspy


def exact_match(example: dspy.Example, prediction, trace=None) -> float:
    """Check if prediction exactly matches expected answer.

    Args:
        example: Ground truth example with 'answer' field
        prediction: Model prediction with 'answer' field
        trace: Optional execution trace (unused)

    Returns:
        1.0 if exact match, 0.0 otherwise

    Example:
        >>> optimizer = MIPROOptimizer(config=config, metric=exact_match)
    """
    if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
        return 0.0

    pred_answer = str(prediction.answer).strip().lower()
    true_answer = str(example.answer).strip().lower()

    return 1.0 if pred_answer == true_answer else 0.0


def contains_match(example: dspy.Example, prediction, trace=None) -> float:
    """Check if prediction contains the expected answer.

    More lenient than exact_match - useful when answer might have extra explanation.

    Args:
        example: Ground truth example with 'answer' field
        prediction: Model prediction with 'answer' field
        trace: Optional execution trace (unused)

    Returns:
        1.0 if answer is contained, 0.0 otherwise

    Example:
        >>> optimizer = MIPROOptimizer(config=config, metric=contains_match)
    """
    if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
        return 0.0

    pred_answer = str(prediction.answer).strip().lower()
    true_answer = str(example.answer).strip().lower()

    return 1.0 if true_answer in pred_answer else 0.0


def numeric_match(example: dspy.Example, prediction, trace=None) -> float:
    """Check if numeric values match (ignores formatting).

    Extracts first number from both prediction and expected answer.

    Args:
        example: Ground truth example with 'answer' field
        prediction: Model prediction with 'answer' field
        trace: Optional execution trace (unused)

    Returns:
        1.0 if numbers match, 0.9 if close, 0.0 otherwise

    Example:
        >>> optimizer = MIPROOptimizer(config=config, metric=numeric_match)
    """
    if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
        return 0.0

    pred_answer = str(prediction.answer).strip()
    true_answer = str(example.answer).strip()

    # Extract numbers
    pred_numbers = re.findall(r"-?\d+\.?\d*", pred_answer)
    true_numbers = re.findall(r"-?\d+\.?\d*", true_answer)

    if not pred_numbers or not true_numbers:
        # Fall back to contains match if no numbers
        return 1.0 if true_answer.lower() in pred_answer.lower() else 0.0

    # Compare first numbers
    try:
        pred_num = float(pred_numbers[0])
        true_num = float(true_numbers[0])

        if pred_num == true_num:
            return 1.0
        elif abs(pred_num - true_num) / max(abs(true_num), 1) < 0.01:  # Within 1%
            return 0.9
        else:
            return 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def make_gepa_metric(simple_metric: Callable, feedback_fn: Callable | None = None) -> Callable:
    """Convert a simple metric (returns float) to GEPA format (returns Prediction with feedback).

    GEPA requires metrics to accept 5 arguments and return dspy.Prediction with score and feedback.
    This helper converts simple metrics to that format.

    Args:
        simple_metric: A metric function that accepts (example, prediction, trace) and returns float
        feedback_fn: Optional function (example, prediction, score) -> str for custom feedback

    Returns:
        A GEPA-compatible metric function

    Example:
        >>> def my_metric(example, pred, trace=None):
        ...     return 1.0 if example.answer == pred.answer else 0.0
        >>>
        >>> gepa_metric = make_gepa_metric(my_metric)
        >>> optimizer = GEPAOptimizer(config=config, metric=gepa_metric)

    Example with custom feedback:
        >>> def custom_feedback(example, prediction, score):
        ...     if score > 0.5:
        ...         return "Good job!"
        ...     return f"Expected '{example.answer}' but got '{prediction.answer}'"
        >>>
        >>> gepa_metric = make_gepa_metric(my_metric, custom_feedback)
    """

    def gepa_metric(
        gold: dspy.Example, pred, trace=None, pred_name=None, pred_trace=None
    ) -> dspy.Prediction:
        # Call the simple metric
        score = simple_metric(gold, pred, trace)

        # Generate feedback
        if feedback_fn:
            feedback = feedback_fn(gold, pred, score)
        else:
            # Default feedback
            if score >= 1.0:
                feedback = "Perfect! Your answer is correct."
            elif score >= 0.8:
                feedback = "Good! Your answer is mostly correct."
            elif score >= 0.5:
                feedback = f"Partial. Expected '{gold.answer}' but got '{pred.answer if hasattr(pred, 'answer') else pred}'"
            else:
                feedback = f"Incorrect. Expected '{gold.answer}' but got '{pred.answer if hasattr(pred, 'answer') else pred}'"

        return dspy.Prediction(score=score, feedback=feedback)

    return gepa_metric


# Pre-made GEPA metrics
exact_match_gepa = make_gepa_metric(exact_match)
contains_match_gepa = make_gepa_metric(contains_match)
numeric_match_gepa = make_gepa_metric(numeric_match)
