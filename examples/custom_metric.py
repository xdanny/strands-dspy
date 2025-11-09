"""
Custom metric example for domain-specific optimization.

This example demonstrates:
1. Creating custom metric functions for MIPRO and GEPA
2. Domain-specific evaluation criteria
3. Using feedback to guide GEPA optimization
4. Comparing MIPRO vs GEPA approaches
"""

import dspy


# Example 1: Simple accuracy metric for MIPRO
def accuracy_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Simple accuracy metric for MIPRO.

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Execution trace (optional)

    Returns:
        Score between 0.0 and 1.0
    """
    # Exact match
    if example.answer.lower().strip() == prediction.answer.lower().strip():
        return 1.0

    # Partial credit for substring match
    if example.answer.lower() in prediction.answer.lower():
        return 0.7

    return 0.0


# Example 2: Semantic similarity metric for MIPRO
def semantic_similarity_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> float:
    """Semantic similarity metric using simple heuristics.

    In production, you might use embedding similarity or
    an LLM-as-judge approach.
    """
    answer_words = set(example.answer.lower().split())
    pred_words = set(prediction.answer.lower().split())

    # Jaccard similarity
    intersection = answer_words.intersection(pred_words)
    union = answer_words.union(pred_words)

    if not union:
        return 0.0

    return len(intersection) / len(union)


# Example 3: Feedback metric for GEPA
def feedback_metric_gepa(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> dspy.Prediction:
    """Rich feedback metric for GEPA.

    GEPA uses textual feedback to guide optimization,
    not just numeric scores.
    """
    answer = example.answer.lower().strip()
    pred = prediction.answer.lower().strip()

    # Check exact match
    if answer == pred:
        return dspy.Prediction(score=1.0, feedback="Perfect match! Answer is completely correct.")

    # Check if answer is contained
    if answer in pred:
        return dspy.Prediction(
            score=0.8,
            feedback=f"Answer contains the correct information ({answer}) but includes extra details.",
        )

    # Check if key terms are present
    answer_terms = set(answer.split())
    pred_terms = set(pred.split())
    overlap = answer_terms.intersection(pred_terms)

    if len(overlap) >= len(answer_terms) * 0.5:
        missing = answer_terms - pred_terms
        return dspy.Prediction(
            score=0.5,
            feedback=f"Partial match. Contains some key terms but missing: {', '.join(missing)}",
        )

    # Completely wrong
    return dspy.Prediction(
        score=0.0,
        feedback=f"Incorrect answer. Expected something related to '{answer}' but got '{pred[:50]}...'",
    )


# Example 4: Multi-criteria metric with detailed feedback
def multi_criteria_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> dspy.Prediction:
    """Evaluate on multiple criteria with detailed feedback.

    This is useful for complex tasks where accuracy alone isn't enough.
    """
    scores = {}
    feedback_parts = []

    # Criterion 1: Correctness
    if example.answer.lower() in prediction.answer.lower():
        scores["correctness"] = 1.0
        feedback_parts.append("✓ Answer is correct")
    else:
        scores["correctness"] = 0.0
        feedback_parts.append("✗ Answer is incorrect")

    # Criterion 2: Conciseness (prefer shorter answers)
    word_count = len(prediction.answer.split())
    if word_count <= 20:
        scores["conciseness"] = 1.0
        feedback_parts.append("✓ Answer is concise")
    elif word_count <= 50:
        scores["conciseness"] = 0.7
        feedback_parts.append("~ Answer is somewhat verbose")
    else:
        scores["conciseness"] = 0.3
        feedback_parts.append("✗ Answer is too verbose")

    # Criterion 3: Completeness (contains key information)
    required_terms = getattr(example, "required_terms", [])
    if required_terms:
        present_terms = [
            term for term in required_terms if term.lower() in prediction.answer.lower()
        ]
        scores["completeness"] = len(present_terms) / len(required_terms)

        if scores["completeness"] == 1.0:
            feedback_parts.append("✓ All required information present")
        else:
            missing = set(required_terms) - set(present_terms)
            feedback_parts.append(f"✗ Missing information: {', '.join(missing)}")
    else:
        scores["completeness"] = 1.0

    # Calculate weighted average
    weights = {"correctness": 0.6, "conciseness": 0.2, "completeness": 0.2}
    total_score = sum(scores[k] * weights[k] for k in scores)

    # Combine feedback
    feedback = "\n".join(feedback_parts)
    feedback += f"\n\nScores: Correctness={scores['correctness']:.1f}, Conciseness={scores['conciseness']:.1f}, Completeness={scores['completeness']:.1f}"

    return dspy.Prediction(score=total_score, feedback=feedback)


# Example 5: LLM-as-judge metric (requires LLM API)
def llm_judge_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> dspy.Prediction:
    """Use an LLM to judge the quality of the answer.

    This is the most flexible approach but requires API calls.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate the quality of an answer."""

        question: str = dspy.InputField()
        reference_answer: str = dspy.InputField()
        candidate_answer: str = dspy.InputField()
        score: float = dspy.OutputField(desc="Score from 0.0 to 1.0")
        feedback: str = dspy.OutputField(desc="Explanation of the score")

    try:
        judge = dspy.Predict(JudgeSignature)
        result = judge(
            question=example.question,
            reference_answer=example.answer,
            candidate_answer=prediction.answer,
        )

        return dspy.Prediction(score=float(result.score), feedback=result.feedback)

    except Exception:
        # Fallback to simple metric if LLM judge fails
        return feedback_metric_gepa(example, prediction, trace)


# Example 6: Task-specific classification metric
def classification_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> dspy.Prediction:
    """Metric for classification tasks.

    Provides specific feedback about classification errors.
    """
    correct_category = example.category.lower().strip()
    pred_category = prediction.category.lower().strip()

    if correct_category == pred_category:
        # Check confidence alignment
        expected_confidence = getattr(example, "confidence", None)
        pred_confidence = getattr(prediction, "confidence", None)

        if expected_confidence and pred_confidence:
            if expected_confidence == pred_confidence:
                return dspy.Prediction(
                    score=1.0,
                    feedback=f"Correct category ({correct_category}) with appropriate confidence ({pred_confidence})",
                )
            else:
                return dspy.Prediction(
                    score=0.9,
                    feedback=f"Correct category ({correct_category}) but confidence mismatch (expected {expected_confidence}, got {pred_confidence})",
                )

        return dspy.Prediction(score=1.0, feedback=f"Correct category: {correct_category}")

    else:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Incorrect category. Expected '{correct_category}' but predicted '{pred_category}'. Review the distinguishing features of these categories.",
        )


# Example usage guide
def print_metric_guide():
    """Print usage guide for custom metrics."""
    print(
        """
Custom Metric Examples for strands-dspy
========================================

1. Simple Accuracy Metric (MIPRO)
   - Use when: Binary correct/incorrect evaluation
   - Returns: float (0.0 to 1.0)
   - Example: accuracy_metric()

2. Semantic Similarity Metric (MIPRO)
   - Use when: Fuzzy matching is acceptable
   - Returns: float (0.0 to 1.0)
   - Example: semantic_similarity_metric()

3. Feedback Metric (GEPA)
   - Use when: You want rich feedback to guide optimization
   - Returns: dspy.Prediction(score, feedback)
   - Example: feedback_metric_gepa()

4. Multi-Criteria Metric (GEPA)
   - Use when: Multiple dimensions of quality matter
   - Returns: dspy.Prediction(score, feedback)
   - Example: multi_criteria_metric()

5. LLM-as-Judge Metric (GEPA)
   - Use when: Automated evaluation is critical
   - Returns: dspy.Prediction(score, feedback)
   - Example: llm_judge_metric()

6. Task-Specific Metric (GEPA)
   - Use when: Domain-specific evaluation criteria
   - Returns: dspy.Prediction(score, feedback)
   - Example: classification_metric()

Integration with strands-dspy
------------------------------

from strands_dspy import DSPyOptimizationHook, OptimizationConfig

# For MIPRO (simpler metrics)
optimizer = DSPyOptimizationHook(
    config=OptimizationConfig(optimizer_type="mipro"),
    metric=accuracy_metric,  # or semantic_similarity_metric
    ...
)

# For GEPA (feedback-based metrics)
optimizer = DSPyOptimizationHook(
    config=OptimizationConfig(optimizer_type="gepa"),
    metric=feedback_metric_gepa,  # or multi_criteria_metric
    ...
)
"""
    )


if __name__ == "__main__":
    print_metric_guide()
