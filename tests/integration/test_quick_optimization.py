"""Quick integration test for MIPRO/GEPA optimization (fast version)."""

import pytest
import dspy
from strands_dspy.optimizers.mipro import MIPROOptimizer
from strands_dspy.optimizers.gepa import GEPAOptimizer
from strands_dspy.types import OptimizationConfig
from tests.fixtures import split_golden_dataset, contains_metric, gepa_feedback_metric
from tests.test_config import setup_test_env


class SimpleQA(dspy.Module):
    """Simple QA program for testing optimization."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate_answer(question=question)
        return dspy.Prediction(answer=result.answer)


@pytest.mark.integration
def test_mipro_quick():
    """Quick MIPRO optimization test with minimal dataset."""

    # Setup DSPy
    try:
        setup_test_env()
    except ValueError as e:
        pytest.skip(f"Skipping test: {e}")

    # Very small dataset for quick test
    train_examples, val_examples = split_golden_dataset(train_ratio=0.7)
    train_examples = train_examples[:5]  # Only 5 training examples
    val_examples = val_examples[:2]      # Only 2 validation examples

    print(f"\n📊 Quick test dataset: {len(train_examples)} train, {len(val_examples)} val")

    # Minimal MIPRO config (auto_budget will determine num_trials automatically)
    config = OptimizationConfig(
        optimizer_type="mipro",
        auto_budget="light",     # Use light budget for quick test
        minibatch=True,
        minibatch_size=2,
    )

    optimizer = MIPROOptimizer(
        config=config,
        metric=contains_metric,
    )

    # Run optimization
    print("\n🚀 Running quick MIPRO optimization...")
    optimized_program, result = optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    print(f"\n✨ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")
    print(f"   Optimized prompts: {list(result.prompts.keys())}")

    # Verify results
    assert result.best_score is not None, "Should have a best score"
    assert result.best_score >= 0, "Score should be non-negative"
    assert len(result.prompts) > 0, "Should have optimized prompts"

    print("\n✅ Quick MIPRO test passed!")


@pytest.mark.integration
def test_gepa_quick():
    """Quick GEPA optimization test with minimal dataset."""

    # Setup DSPy
    try:
        setup_test_env()
    except ValueError as e:
        pytest.skip(f"Skipping test: {e}")

    # Very small dataset
    train_examples, val_examples = split_golden_dataset(train_ratio=0.7)
    train_examples = train_examples[:5]
    val_examples = val_examples[:2]

    print(f"\n📊 Quick test dataset: {len(train_examples)} train, {len(val_examples)} val")

    # Minimal GEPA config
    config = OptimizationConfig(
        optimizer_type="gepa",
        auto_budget="light",     # Use light budget for quick test
        minibatch_size=2,
        track_stats=True,
    )

    optimizer = GEPAOptimizer(
        config=config,
        metric=gepa_feedback_metric,
    )

    # Run optimization
    print("\n🚀 Running quick GEPA optimization...")
    optimized_program, result = optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    print(f"\n✨ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")
    print(f"   Optimized prompts: {list(result.prompts.keys())}")

    # Verify results
    assert result.best_score is not None, "Should have a best score"
    assert result.best_score >= 0, "Score should be non-negative"
    assert len(result.prompts) > 0, "Should have optimized prompts"

    print("\n✅ Quick GEPA test passed!")


if __name__ == "__main__":
    # Run tests directly
    print("=" * 70)
    print("Quick Optimization Integration Tests")
    print("=" * 70)

    test_mipro_quick()
    print("\n" + "=" * 70)
    test_gepa_quick()
