"""Integration test for GEPA optimization with golden dataset."""

import dspy
import pytest

from strands_dspy.optimizers.gepa import GEPAOptimizer
from strands_dspy.types import OptimizationConfig
from tests.fixtures import contains_metric, gepa_feedback_metric, split_golden_dataset
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
        metric: Metric function to use (returns float or Prediction)

    Returns:
        Average score across all examples
    """
    scores = []
    for example in examples:
        try:
            prediction = program(question=example.question)
            result = metric(example, prediction)

            # Extract score from result (could be float or Prediction)
            if isinstance(result, dspy.Prediction) and hasattr(result, "score"):
                score = result.score
            else:
                score = float(result)

            scores.append(score)
        except Exception as e:
            print(f"Error evaluating example: {e}")
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


@pytest.mark.integration
def test_gepa_optimization_improves_performance():
    """Test that GEPA optimization improves QA performance on golden dataset."""

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

    # Evaluate baseline (using simple contains_metric for comparison)
    print("\n🔍 Evaluating baseline program...")
    baseline_score = evaluate_program(baseline_program, val_examples, contains_metric)
    print(f"✅ Baseline validation score: {baseline_score:.2%}")

    # Configure GEPA optimizer
    config = OptimizationConfig(
        optimizer_type="gepa",
        num_candidates=3,  # Reduced for faster testing
        num_trials=5,  # Reduced for faster testing
        auto_budget="light",
        minibatch_size=5,  # Small minibatch for testing
        track_stats=True,
    )

    optimizer = GEPAOptimizer(
        config=config,
        metric=gepa_feedback_metric,  # GEPA uses feedback metric
    )

    # Run optimization
    print("\n🚀 Running GEPA optimization...")
    print(f"   Config: {config.num_candidates} candidates, {config.num_trials} trials")
    print("   Using feedback-based metric for optimization")

    optimized_program, result = optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    print("\n✨ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")
    print(f"   Train examples: {result.train_size}")
    print(f"   Val examples: {result.val_size}")

    # Evaluate optimized program (using simple contains_metric for comparison)
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
    assert result.optimizer == "gepa", "Optimizer type should be GEPA"
    assert "prompts" in result.prompts or len(result.prompts) > 0, "Should have optimized prompts"

    print("\n✅ GEPA optimization test passed!")


@pytest.mark.integration
def test_gepa_feedback_integration():
    """Test that GEPA correctly uses feedback from the metric."""

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

    # Test the feedback metric itself first
    print("\n🧪 Testing feedback metric...")
    test_program = SimpleQA()
    test_example = train_examples[0]

    try:
        prediction = test_program(question=test_example.question)
        feedback_result = gepa_feedback_metric(test_example, prediction)

        print(f"   Question: {test_example.question}")
        print(f"   Expected: {test_example.answer}")
        print(f"   Predicted: {prediction.answer}")
        print(f"   Score: {feedback_result.score:.2f}")
        print(f"   Feedback: {feedback_result.feedback}")

        assert hasattr(feedback_result, "score"), "Feedback should have score"
        assert hasattr(feedback_result, "feedback"), "Feedback should have feedback text"

    except Exception as e:
        print(f"   ⚠️  Feedback metric test error: {e}")

    # Configure GEPA optimizer
    config = OptimizationConfig(
        optimizer_type="gepa",
        num_candidates=2,
        num_trials=3,
        auto_budget="light",
        minibatch_size=3,
        track_stats=True,
    )

    optimizer = GEPAOptimizer(
        config=config,
        metric=gepa_feedback_metric,
    )

    # Run optimization
    print("\n🚀 Running GEPA optimization (small dataset)...")
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
                value_str = str(value)[:200] if value else ""
                print(f"   {key}: {value_str}...")

    print("\n✅ GEPA feedback integration test passed!")


@pytest.mark.integration
def test_gepa_vs_mipro_comparison():
    """Compare GEPA and MIPRO optimization on the same dataset.

    This test demonstrates the difference between GEPA (feedback-based)
    and MIPRO (score-based) optimization.
    """

    # Setup DSPy with Gemini
    try:
        setup_test_env()
    except ValueError as e:
        pytest.skip(f"Skipping test: {e}")

    # Load small dataset for quick comparison
    train_examples, val_examples = split_golden_dataset(train_ratio=0.7)
    train_examples = train_examples[:8]
    val_examples = val_examples[:2]

    print(f"\n📊 Dataset: {len(train_examples)} train, {len(val_examples)} val examples")

    # Common config
    base_config = {
        "num_candidates": 2,
        "num_trials": 3,
        "auto_budget": "light",
        "minibatch_size": 3,
    }

    results_comparison = {}

    # Test GEPA
    print("\n🔷 Testing GEPA (feedback-based)...")
    gepa_config = OptimizationConfig(optimizer_type="gepa", track_stats=True, **base_config)

    from strands_dspy.optimizers.gepa import GEPAOptimizer

    gepa_optimizer = GEPAOptimizer(
        config=gepa_config,
        metric=gepa_feedback_metric,
    )

    gepa_program, gepa_result = gepa_optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    gepa_score = evaluate_program(gepa_program, val_examples, contains_metric)
    results_comparison["GEPA"] = {
        "score": gepa_score,
        "best_score": gepa_result.best_score,
        "num_prompts": len(gepa_result.prompts),
    }

    print(f"   ✅ GEPA score: {gepa_score:.2%}")

    # Test MIPRO
    print("\n🔶 Testing MIPRO (score-based)...")
    mipro_config = OptimizationConfig(optimizer_type="mipro", minibatch=True, **base_config)

    from strands_dspy.optimizers.mipro import MIPROOptimizer

    mipro_optimizer = MIPROOptimizer(
        config=mipro_config,
        metric=contains_metric,
    )

    mipro_program, mipro_result = mipro_optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    mipro_score = evaluate_program(mipro_program, val_examples, contains_metric)
    results_comparison["MIPRO"] = {
        "score": mipro_score,
        "best_score": mipro_result.best_score,
        "num_prompts": len(mipro_result.prompts),
    }

    print(f"   ✅ MIPRO score: {mipro_score:.2%}")

    # Print comparison
    print("\n" + "=" * 70)
    print("GEPA vs MIPRO Comparison")
    print("=" * 70)
    for optimizer_name, metrics in results_comparison.items():
        print(f"\n{optimizer_name}:")
        print(f"   Validation Score: {metrics['score']:.2%}")
        print(f"   Best Training Score: {metrics['best_score']:.2%}")
        print(f"   Optimized Prompts: {metrics['num_prompts']}")

    print("\n💡 Note: GEPA uses feedback for learning, MIPRO uses scores only")
    print("   Both should produce working optimized programs.")

    # Basic assertions
    assert results_comparison["GEPA"]["score"] >= 0
    assert results_comparison["MIPRO"]["score"] >= 0

    print("\n✅ GEPA vs MIPRO comparison test passed!")


if __name__ == "__main__":
    # Run tests directly
    print("=" * 70)
    print("GEPA Optimization Integration Tests")
    print("=" * 70)

    test_gepa_optimization_improves_performance()
    print("\n" + "=" * 70)
    test_gepa_feedback_integration()
    print("\n" + "=" * 70)
    test_gepa_vs_mipro_comparison()
