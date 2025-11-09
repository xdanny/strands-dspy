"""Harder test dataset that requires actual optimization to solve well."""

import dspy

# Harder reasoning problems that benefit from optimization
HARD_REASONING_PAIRS = [
    {
        "question": "If a train leaves station A at 60 mph and another leaves station B (120 miles away) at 40 mph heading toward each other, when will they meet?",
        "answer": "1.2 hours",
        "reasoning_required": True,
    },
    {
        "question": "A farmer has chickens and cows. He counts 30 heads and 74 legs. How many chickens does he have?",
        "answer": "23 chickens",
        "reasoning_required": True,
    },
    {
        "question": "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
        "answer": "42",
        "reasoning_required": True,
    },
    {
        "question": "If you have a 3-gallon jug and a 5-gallon jug, how can you measure exactly 4 gallons?",
        "answer": "Fill 5-gallon jug, pour into 3-gallon (leaving 2), empty 3-gallon, pour the 2 gallons in, fill 5-gallon again, pour into 3-gallon (which has 2, so only 1 goes in), leaving 4 in the 5-gallon jug",
        "reasoning_required": True,
    },
    {
        "question": "A clock shows 3:15. What is the angle between the hour and minute hands?",
        "answer": "7.5 degrees",
        "reasoning_required": True,
    },
    {
        "question": "You have 12 coins, one is counterfeit (lighter). Using a balance scale only 3 times, how do you find it?",
        "answer": "Divide into groups of 4, weigh two groups. If equal, counterfeit is in third group. Continue subdividing the suspect group.",
        "reasoning_required": True,
    },
    {
        "question": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
        "answer": "Yes",
        "reasoning_required": True,
    },
    {
        "question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "answer": "$0.05",
        "reasoning_required": True,
    },
    {
        "question": "If 5 machines make 5 widgets in 5 minutes, how long would it take 100 machines to make 100 widgets?",
        "answer": "5 minutes",
        "reasoning_required": True,
    },
    {
        "question": "What is the sum of all numbers from 1 to 100?",
        "answer": "5050",
        "reasoning_required": True,
    },
]


def get_hard_dataset() -> list[dspy.Example]:
    """Get hard reasoning dataset as DSPy examples.

    Returns:
        List of DSPy examples with inputs marked
    """
    examples = []

    for item in HARD_REASONING_PAIRS:
        ex = dspy.Example(
            question=item["question"],
            answer=item["answer"],
        ).with_inputs("question")
        examples.append(ex)

    return examples


def split_hard_dataset(
    train_ratio: float = 0.7,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Split hard dataset into train and validation sets.

    Args:
        train_ratio: Fraction of data to use for training (default: 0.7)

    Returns:
        Tuple of (train_examples, val_examples)
    """
    examples = get_hard_dataset()

    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    return train_examples, val_examples


def hard_reasoning_metric(
    gold: dspy.Example, pred, trace=None, pred_name=None, pred_trace=None
) -> dspy.Prediction:
    """Strict metric for hard reasoning problems with detailed feedback.

    This metric is much stricter and provides rich feedback for GEPA.

    Args:
        gold: Ground truth example
        pred: Model prediction
        trace: Optional execution trace
        pred_name: Name of the predictor
        pred_trace: Detailed predictor trace

    Returns:
        dspy.Prediction with score and feedback fields
    """
    if not hasattr(pred, "answer") or not hasattr(gold, "answer"):
        return dspy.Prediction(
            score=0.0,
            feedback="ERROR: Missing answer field. Your response must include an 'answer' field with the final answer.",
        )

    pred_answer = str(pred.answer).strip().lower()
    true_answer = str(gold.answer).strip().lower()

    # Extract numbers from answers for numeric comparison
    import re

    pred_numbers = re.findall(r"-?\d+\.?\d*", pred_answer)
    true_numbers = re.findall(r"-?\d+\.?\d*", true_answer)

    # Exact match (best)
    if pred_answer == true_answer:
        return dspy.Prediction(
            score=1.0,
            feedback="PERFECT: Your answer exactly matches the expected answer. Great work!",
        )

    # Numeric match (for math problems)
    if pred_numbers and true_numbers and pred_numbers[0] == true_numbers[0]:
        return dspy.Prediction(
            score=0.9,
            feedback=f"GOOD: Your numerical answer ({pred_numbers[0]}) is correct, but the format doesn't exactly match. Expected format: '{true_answer}'",
        )

    # Partial match (contains key elements)
    if true_answer in pred_answer or pred_answer in true_answer:
        return dspy.Prediction(
            score=0.6,
            feedback=f"PARTIAL: Your answer contains some correct elements but is not precise enough. You said '{pred_answer}' but the expected answer is '{true_answer}'. Be more specific.",
        )

    # Check if reasoning is present (even if answer is wrong)
    has_reasoning = hasattr(pred, "reasoning") and pred.reasoning and len(pred.reasoning) > 50

    if has_reasoning:
        return dspy.Prediction(
            score=0.3,
            feedback=f"WRONG ANSWER: You provided reasoning which is good, but your answer '{pred_answer}' is incorrect. The correct answer is '{true_answer}'. Review your reasoning - you may have made a calculation error or logical mistake. Focus on: 1) Breaking down the problem step-by-step, 2) Double-checking calculations, 3) Stating the final answer clearly.",
        )
    else:
        return dspy.Prediction(
            score=0.1,
            feedback=f"INCOMPLETE: Your answer '{pred_answer}' is wrong AND you didn't show your reasoning. The correct answer is '{true_answer}'. You MUST: 1) Think through the problem step-by-step in the 'reasoning' field, 2) Show your work and calculations, 3) State the final answer clearly. Without reasoning, I can't help you improve.",
        )


def simple_contains_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Simple metric for MIPRO (returns float)."""
    if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
        return 0.0

    pred_answer = str(prediction.answer).strip().lower()
    true_answer = str(example.answer).strip().lower()

    # Extract numbers
    import re

    pred_numbers = re.findall(r"-?\d+\.?\d*", pred_answer)
    true_numbers = re.findall(r"-?\d+\.?\d*", true_answer)

    # Exact match
    if pred_answer == true_answer:
        return 1.0

    # Numeric match
    if pred_numbers and true_numbers and pred_numbers[0] == true_numbers[0]:
        return 0.9

    # Contains
    if true_answer in pred_answer:
        return 0.6

    return 0.0
