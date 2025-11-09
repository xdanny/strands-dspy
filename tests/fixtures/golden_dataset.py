"""Golden test dataset for DSPy optimization testing.

This module provides curated Q&A pairs with known correct answers,
used for testing prompt optimization quality.
"""

import dspy

# Golden dataset: Well-defined Q&A pairs with clear correct answers
GOLDEN_QA_PAIRS = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": "France is a country in Western Europe.",
    },
    {
        "question": "What is 2 + 2?",
        "answer": "4",
        "context": "Basic arithmetic problem.",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare",
        "context": "Classic English literature.",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
        "context": "Planetary science.",
    },
    {
        "question": "What is the chemical symbol for water?",
        "answer": "H2O",
        "context": "Basic chemistry.",
    },
    {
        "question": "In what year did World War II end?",
        "answer": "1945",
        "context": "World War II history.",
    },
    {
        "question": "What is the speed of light in vacuum?",
        "answer": "299,792,458 meters per second",
        "context": "Physics - fundamental constants.",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "answer": "Leonardo da Vinci",
        "context": "Art history - Renaissance period.",
    },
    {
        "question": "What is the smallest prime number?",
        "answer": "2",
        "context": "Mathematics - number theory.",
    },
    {
        "question": "What is the capital of Japan?",
        "answer": "Tokyo",
        "context": "Geography - East Asia.",
    },
    {
        "question": "How many continents are there?",
        "answer": "7",
        "context": "Geography - world continents.",
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "answer": "100 degrees Celsius",
        "context": "Physics - phase transitions.",
    },
    {
        "question": "Who developed the theory of relativity?",
        "answer": "Albert Einstein",
        "context": "Physics - modern physics.",
    },
    {
        "question": "What is the largest ocean on Earth?",
        "answer": "Pacific Ocean",
        "context": "Geography - oceanography.",
    },
    {
        "question": "How many sides does a hexagon have?",
        "answer": "6",
        "context": "Mathematics - geometry.",
    },
    {
        "question": "What is the primary gas in Earth's atmosphere?",
        "answer": "Nitrogen",
        "context": "Chemistry - atmospheric composition.",
    },
    {
        "question": "Who was the first person to walk on the Moon?",
        "answer": "Neil Armstrong",
        "context": "Space exploration - Apollo 11.",
    },
    {
        "question": "What is the square root of 144?",
        "answer": "12",
        "context": "Mathematics - square roots.",
    },
    {
        "question": "What is the hardest natural substance on Earth?",
        "answer": "Diamond",
        "context": "Mineralogy - material properties.",
    },
    {
        "question": "How many planets are in our solar system?",
        "answer": "8",
        "context": "Astronomy - solar system.",
    },
]


def get_golden_dataset(with_context: bool = False) -> list[dspy.Example]:
    """Get golden dataset as DSPy examples.

    Args:
        with_context: If True, include context field in examples

    Returns:
        List of DSPy examples with inputs marked
    """
    examples = []

    for item in GOLDEN_QA_PAIRS:
        if with_context:
            ex = dspy.Example(
                question=item["question"],
                answer=item["answer"],
                context=item["context"],
            ).with_inputs("question", "context")
        else:
            ex = dspy.Example(
                question=item["question"],
                answer=item["answer"],
            ).with_inputs("question")

        examples.append(ex)

    return examples


def split_golden_dataset(
    train_ratio: float = 0.7,
    with_context: bool = False,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Split golden dataset into train and validation sets.

    Args:
        train_ratio: Fraction of data to use for training (default: 0.7)
        with_context: If True, include context field in examples

    Returns:
        Tuple of (train_examples, val_examples)
    """
    examples = get_golden_dataset(with_context=with_context)

    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    return train_examples, val_examples


# Metrics for evaluating QA performance
def exact_match_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Exact match metric - checks if prediction exactly matches answer.

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional execution trace

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
        return 0.0

    pred_answer = str(prediction.answer).strip().lower()
    true_answer = str(example.answer).strip().lower()

    return 1.0 if pred_answer == true_answer else 0.0


def contains_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Contains metric - checks if prediction contains the answer.

    More lenient than exact match - useful for longer explanations.

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional execution trace

    Returns:
        1.0 if answer is contained in prediction, 0.0 otherwise
    """
    if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
        return 0.0

    pred_answer = str(prediction.answer).strip().lower()
    true_answer = str(example.answer).strip().lower()

    return 1.0 if true_answer in pred_answer else 0.0


def gepa_feedback_metric(
    gold: dspy.Example, pred, trace=None, pred_name=None, pred_trace=None
) -> dspy.Prediction:
    """Feedback metric for GEPA optimization.

    Returns both a score and textual feedback for the optimizer to learn from.

    GEPA requires metrics to accept 5 arguments as per the new API:
    https://dspy.ai/api/optimizers/GEPA

    Args:
        gold: Ground truth example
        pred: Model prediction
        trace: Optional execution trace
        pred_name: Name of the predictor (optional)
        pred_trace: Detailed predictor trace (optional)

    Returns:
        dspy.Prediction with score and feedback fields
    """
    if not hasattr(pred, "answer") or not hasattr(gold, "answer"):
        return dspy.Prediction(score=0.0, feedback="Missing answer field in prediction or example")

    pred_answer = str(pred.answer).strip().lower()
    true_answer = str(gold.answer).strip().lower()

    # Exact match
    if pred_answer == true_answer:
        return dspy.Prediction(
            score=1.0, feedback="Perfect! The answer exactly matches the expected answer."
        )

    # Contains correct answer
    if true_answer in pred_answer:
        return dspy.Prediction(
            score=0.8,
            feedback=f"Good! The answer contains the correct information '{true_answer}', but has extra text.",
        )

    # Partially correct (simple heuristic)
    words_true = set(true_answer.split())
    words_pred = set(pred_answer.split())
    overlap = len(words_true & words_pred) / max(len(words_true), 1)

    if overlap > 0.5:
        return dspy.Prediction(
            score=0.5,
            feedback=f"Partially correct. Expected '{true_answer}' but got '{pred_answer}'. Some keywords match.",
        )

    # Completely wrong
    return dspy.Prediction(
        score=0.0, feedback=f"Incorrect. Expected '{true_answer}' but got '{pred_answer}'."
    )
