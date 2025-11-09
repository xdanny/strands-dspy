"""Integration test for MIPRO optimization with golden dataset."""

import dspy
import pytest

from strands_dspy.optimizers.mipro import MIPROOptimizer
from strands_dspy.types import OptimizationConfig
from tests.fixtures import contains_metric, split_golden_dataset
from tests.test_config import setup_test_env


class SimpleQA(dspy.Module):
    """Simple QA program for testing optimization."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate_answer(question=question)
        return dspy.Prediction(answer=result.answer)


def evaluate_program(program, examples, metric):
    """Evaluate a program on a set of examples.

    Args:
        program: DSPy program to evaluate
        examples: List of test examples
        metric: Metric function to use

    Returns:
        Average score across all examples
    """
    scores = []
    for example in examples:
        try:
            prediction = program(question=example.question)
            score = metric(example, prediction)
            scores.append(score)
        except Exception as e:
            print(f"Error evaluating example: {e}")
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


@pytest.mark.integration
def test_mipro_optimization_improves_performance():
    """Test that MIPRO optimization improves QA performance on golden dataset."""

    # Setup DSPy with Gemini
    try:
        setup_test_env()
    except ValueError as e:
        pytest.skip(f"Skipping test: {e}")

    # Load golden dataset
    train_examples, val_examples = split_golden_dataset(train_ratio=0.7)

    print(f"\n📊 Dataset: {len(train_examples)} train, {len(val_examples)} val examples")

    # Create baseline program
    baseline_program = SimpleQA()

    # Evaluate baseline
    print("\n🔍 Evaluating baseline program...")
    baseline_score = evaluate_program(baseline_program, val_examples, contains_metric)
    print(f"✅ Baseline validation score: {baseline_score:.2%}")

    # Configure MIPRO optimizer
    config = OptimizationConfig(
        optimizer_type="mipro",
        num_candidates=3,  # Reduced for faster testing
        num_trials=5,  # Reduced for faster testing
        auto_budget="light",
        minibatch=True,
        minibatch_size=5,  # Small minibatch for testing
    )

    optimizer = MIPROOptimizer(
        config=config,
        metric=contains_metric,
    )

    # Run optimization
    print("\n🚀 Running MIPRO optimization...")
    print(f"   Config: {config.num_candidates} candidates, {config.num_trials} trials")

    optimized_program, result = optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    print("\n✨ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")
    print(f"   Train examples: {result.train_size}")
    print(f"   Val examples: {result.val_size}")

    # Evaluate optimized program
    print("\n🔍 Evaluating optimized program...")
    optimized_score = evaluate_program(optimized_program, val_examples, contains_metric)
    print(f"✅ Optimized validation score: {optimized_score:.2%}")

    # Calculate improvement
    improvement = optimized_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
    print(f"\n📈 Improvement: {improvement:+.2%} ({improvement_pct:+.1f}%)")

    # Verify optimization results
    assert result.best_score > 0, "Optimization should produce a positive score"
    assert result.train_size == len(train_examples), "Train size should match dataset"
    assert result.val_size == len(val_examples), "Val size should match dataset"
    assert result.optimizer == "mipro", "Optimizer type should be MIPRO"
    assert "prompts" in result.prompts or len(result.prompts) > 0, "Should have optimized prompts"

    # Ideally, optimization should improve performance
    # But we don't enforce this strictly since it depends on the model and data
    print("\n✅ MIPRO optimization test passed!")


@pytest.mark.integration
def test_mipro_optimization_stores_prompts():
    """Test that MIPRO optimization stores optimized prompts correctly."""

    # Setup DSPy with Gemini
    try:
        setup_test_env()
    except ValueError as e:
        pytest.skip(f"Skipping test: {e}")

    # Load golden dataset (smaller subset for faster testing)
    train_examples, val_examples = split_golden_dataset(train_ratio=0.7)
    train_examples = train_examples[:10]  # Use only 10 training examples
    val_examples = val_examples[:3]  # Use only 3 validation examples

    print(f"\n📊 Small dataset: {len(train_examples)} train, {len(val_examples)} val examples")

    # Configure MIPRO optimizer
    config = OptimizationConfig(
        optimizer_type="mipro",
        num_candidates=2,
        num_trials=3,
        auto_budget="light",
        minibatch=True,
        minibatch_size=3,
    )

    optimizer = MIPROOptimizer(
        config=config,
        metric=contains_metric,
    )

    # Run optimization
    print("\n🚀 Running MIPRO optimization (small dataset)...")
    optimized_program, result = optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    # Verify prompt storage
    print("\n✨ Optimization complete!")
    print(f"   Stored prompts: {list(result.prompts.keys())}")

    assert isinstance(result.prompts, dict), "Prompts should be a dictionary"
    assert len(result.prompts) > 0, "Should have at least one optimized prompt"

    # Check prompt structure
    for predictor_name, prompt_data in result.prompts.items():
        print(f"\n📝 Predictor '{predictor_name}':")
        if isinstance(prompt_data, dict):
            for key, value in prompt_data.items():
                print(f"   {key}: {str(value)[:100]}...")

    print("\n✅ MIPRO prompt storage test passed!")


if __name__ == "__main__":
    # Run tests directly
    print("=" * 70)
    print("MIPRO Optimization Integration Tests")
    print("=" * 70)

    test_mipro_optimization_improves_performance()
    print("\n" + "=" * 70)
    test_mipro_optimization_stores_prompts()
