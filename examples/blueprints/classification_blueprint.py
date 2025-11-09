"""Blueprint: Text Classification Agent with DSPy Optimization

This blueprint shows how to optimize a classification agent (e.g., sentiment
analysis, intent detection, category labeling) using MIPRO optimizer.

Usage:
    1. Copy this file to your project
    2. Update TRAINING_DATA with your labeled examples
    3. Configure your LM
    4. Run optimization

Example:
    $ uv run python classification_blueprint.py
"""

import os

import dspy
from dotenv import load_dotenv
from strands import Agent

from strands_dspy import (
    MIPROOptimizer,
    SessionStorageBackend,
    TrainingCollector,
)
from strands_dspy.helpers import (
    end_turn_success,
    exact_match,
    extract_first_user_text,
    extract_last_assistant_text,
)
from strands_dspy.types import OptimizationConfig

# ============================================================================
# 1. Configure your LM
# ============================================================================

load_dotenv()

lm = dspy.LM(
    model="openrouter/minimax/minimax-m2:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0,
)

dspy.configure(lm=lm)


# ============================================================================
# 2. Define your training data
# ============================================================================

# Example: Sentiment classification
TRAINING_DATA = [
    {"text": "This product is amazing! Best purchase ever!", "label": "positive"},
    {"text": "Terrible quality. Very disappointed.", "label": "negative"},
    {"text": "It's okay, nothing special.", "label": "neutral"},
    {"text": "Absolutely love it! Highly recommend.", "label": "positive"},
    {"text": "Waste of money. Doesn't work as advertised.", "label": "negative"},
    {"text": "Pretty average. Does the job.", "label": "neutral"},
    {"text": "Exceeded my expectations! 5 stars!", "label": "positive"},
    {"text": "Horrible customer service and poor quality.", "label": "negative"},
    {"text": "It's fine. Nothing to complain about.", "label": "neutral"},
    {"text": "Best in class! Couldn't be happier!", "label": "positive"},
    # Add more examples...
]


# ============================================================================
# 3. Create your agent
# ============================================================================

classifier_agent = Agent(
    name="sentiment-classifier",
    instructions="""You are a sentiment analysis assistant.
Classify text into one of these categories: positive, negative, neutral.
Only respond with the label, nothing else.""",
    model="gemini/gemini-2.5-flash-lite",
)


# ============================================================================
# 4. Set up training collection
# ============================================================================

storage = SessionStorageBackend()

collector = TrainingCollector(
    storage=storage,
    input_extractor=extract_first_user_text,
    output_extractor=extract_last_assistant_text,
    success_criteria=end_turn_success,
)

collector.attach(classifier_agent)


# ============================================================================
# 5. Collect training examples
# ============================================================================


def collect_training_examples():
    """Run agent on training data to collect examples."""
    print(f"\n📊 Collecting training examples from {len(TRAINING_DATA)} labeled texts...")

    for i, example in enumerate(TRAINING_DATA):
        text = example["text"]
        print(f"  [{i+1}/{len(TRAINING_DATA)}] {text[:50]}...")

        # Run agent
        classifier_agent.invoke(text)

    # Retrieve examples
    examples = storage.get_training_examples("session-1", "sentiment-classifier")
    print(f"\n✅ Collected {len(examples)} training examples")

    return examples


# ============================================================================
# 6. Define your DSPy program
# ============================================================================


class ClassificationProgram(dspy.Module):
    """Classification DSPy program."""

    def __init__(self):
        super().__init__()
        # For classification, we want structured output
        self.classifier = dspy.ChainOfThought("text -> label")

    def forward(self, text):
        result = self.classifier(text=text)
        return result


# ============================================================================
# 7. Create validation examples
# ============================================================================

VALIDATION_DATA = [
    {"text": "Outstanding product! Will buy again!", "label": "positive"},
    {"text": "Complete garbage. Save your money.", "label": "negative"},
    {"text": "It's alright. Average quality.", "label": "neutral"},
]


# ============================================================================
# 8. Define custom metric for classification
# ============================================================================


def classification_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Metric for classification tasks.

    Uses exact_match helper but could be extended for:
    - Multi-class accuracy
    - F1 score
    - Precision/Recall
    """
    # Use the pre-built exact_match helper
    score = exact_match(example, prediction, trace)

    # Optional: Add partial credit for similar labels
    # if not score and hasattr(prediction, "label") and hasattr(example, "label"):
    #     pred_label = str(prediction.label).strip().lower()
    #     true_label = str(example.label).strip().lower()
    #
    #     # Example: Both positive/negative vs neutral
    #     if pred_label in ["positive", "negative"] and true_label in ["positive", "negative"]:
    #         return 0.3  # Partial credit

    return score


# ============================================================================
# 9. Run MIPRO optimization
# ============================================================================


def run_optimization():
    """Run MIPRO optimization for classification."""

    # Collect examples
    training_examples = collect_training_examples()

    # Convert to DSPy format
    trainset = [
        dspy.Example(text=ex.inputs["question"], label=ex.outputs["answer"]).with_inputs("text")
        for ex in training_examples
    ]

    valset = [
        dspy.Example(text=ex["text"], label=ex["label"]).with_inputs("text")
        for ex in VALIDATION_DATA
    ]

    print("\n🔧 Starting MIPRO optimization...")
    print(f"   Training examples: {len(trainset)}")
    print(f"   Validation examples: {len(valset)}")

    # Configure optimization
    config = OptimizationConfig(
        optimizer_type="mipro",
        auto_budget="medium",
        track_stats=True,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    # Create optimizer
    optimizer = MIPROOptimizer(
        config=config,
        metric=classification_metric,
    )

    # Run optimization
    program = ClassificationProgram()
    optimized_program, result = optimizer.optimize(
        program=program,
        trainset=trainset,
        valset=valset,
    )

    print("\n✨ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")

    # Show optimized prompts
    print("\n📝 Optimized prompts:")
    for predictor_name, prompt_data in result.prompts.items():
        print(f"\n{predictor_name}:")
        if "instruction" in prompt_data:
            print(f"  Instruction: {prompt_data['instruction'][:100]}...")
        if "demos" in prompt_data and prompt_data["demos"]:
            print(f"  Demonstrations: {len(prompt_data['demos'])} examples")

    return optimized_program, result


# ============================================================================
# 10. Test optimized program
# ============================================================================


def test_optimized_program(program):
    """Test the optimized classifier."""

    test_texts = [
        "Incredible value for money! Super happy!",
        "Broke after one use. Total junk.",
        "Does what it says on the box.",
    ]

    print("\n🧪 Testing optimized classifier...")

    for text in test_texts:
        prediction = program(text=text)
        print(f"\n  Text: {text}")
        print(f"  Label: {prediction.label}")


# ============================================================================
# Advanced: Custom extractors for multi-field classification
# ============================================================================


def extract_classification_input(messages):
    """
    Custom extractor for classification with context.

    Example use case: Classification with additional context fields
    like user history, metadata, etc.
    """
    from strands_dspy.helpers import extract_first_user_text

    # Start with basic text extraction
    result = extract_first_user_text(messages)

    # Could add more fields here from metadata
    # result["context"] = extract_context_from_messages(messages)
    # result["user_history"] = extract_user_history(messages)

    return result


# ============================================================================
# Advanced: Multi-label classification metric
# ============================================================================


def multi_label_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Metric for multi-label classification.

    Example: ["positive", "urgent"] vs ["positive", "spam"]
    """
    if not hasattr(prediction, "labels") or not hasattr(example, "labels"):
        return 0.0

    pred_labels = set(str(prediction.labels).split(","))
    true_labels = set(str(example.labels).split(","))

    # Jaccard similarity
    intersection = len(pred_labels & true_labels)
    union = len(pred_labels | true_labels)

    return intersection / union if union > 0 else 0.0


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run optimization
    optimized_program, result = run_optimization()

    # Test it
    test_optimized_program(optimized_program)

    print("\n✅ Done!")

    # Next steps:
    print("\n💡 Next steps:")
    print("  1. Add more training examples to improve accuracy")
    print("  2. Tune OptimizationConfig (try 'heavy' budget)")
    print("  3. Experiment with different metrics")
    print("  4. Try GEPA optimizer for feedback-based improvement")
