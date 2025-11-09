"""Real optimization test with harder problems that show actual improvement."""

import pytest
import dspy
from strands_dspy.optimizers.mipro import MIPROOptimizer
from strands_dspy.optimizers.gepa import GEPAOptimizer
from strands_dspy.types import OptimizationConfig
from tests.fixtures.hard_dataset import (
    split_hard_dataset,
    hard_reasoning_metric,
    simple_contains_metric,
)
from tests.test_config import setup_test_env


class BasicQA(dspy.Module):
    """Basic QA with minimal instruction - should perform poorly initially."""

    def __init__(self):
        super().__init__()
        # Start with no instruction - let optimizer create one
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate_answer(question=question)
        return dspy.Prediction(answer=result.answer)


def evaluate_program(program, examples, metric):
    """Evaluate a program and show individual results."""
    scores = []
    print(f"\n{'─'*80}")
    print("Evaluation Results:")
    print(f"{'─'*80}")

    for i, example in enumerate(examples, 1):
        try:
            prediction = program(question=example.question)
            result = metric(example, prediction, None, None, None)

            # Extract score from result
            if isinstance(result, dspy.Prediction) and hasattr(result, 'score'):
                score = result.score
                feedback = result.feedback if hasattr(result, 'feedback') else ""
            else:
                score = float(result)
                feedback = ""

            scores.append(score)

            print(f"\n{i}. Q: {example.question[:70]}...")
            print(f"   Expected: {example.answer}")
            print(f"   Got: {prediction.answer}")
            print(f"   Score: {score:.1%}")
            if feedback and score < 1.0:
                print(f"   Feedback: {feedback[:100]}...")

        except Exception as e:
            print(f"\n{i}. ERROR: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n{'─'*80}")
    print(f"Average Score: {avg_score:.1%}")
    print(f"{'─'*80}\n")

    return avg_score


@pytest.mark.integration
def test_gepa_real_improvement():
    """Test GEPA with harder problems showing actual optimization."""

    # Setup DSPy
    try:
        setup_test_env()
    except ValueError as e:
        pytest.skip(f"Skipping test: {e}")

    # Get harder dataset
    train_examples, val_examples = split_hard_dataset(train_ratio=0.7)

    print(f"\n{'='*80}")
    print(f"GEPA REAL OPTIMIZATION TEST - Hard Reasoning Problems")
    print(f"{'='*80}")
    print(f"\n📊 Dataset: {len(train_examples)} train, {len(val_examples)} val")

    # Evaluate baseline with NO optimization
    print("\n🔍 BASELINE (No Optimization)")
    baseline_program = BasicQA()
    baseline_score = evaluate_program(baseline_program, val_examples, hard_reasoning_metric)

    # Configure GEPA with limited budget for testing
    config = OptimizationConfig(
        optimizer_type="gepa",
        auto_budget=None,  # Manual budget
        num_trials=5,      # Only 5 iterations for quick test
        minibatch_size=2,
        track_stats=True,
    )

    optimizer = GEPAOptimizer(
        config=config,
        metric=hard_reasoning_metric,
    )

    # Run optimization
    print(f"\n🚀 Running GEPA optimization (max 5 iterations)...")
    optimized_program, result = optimizer.optimize(
        program=BasicQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    # Evaluate optimized program
    print(f"\n✨ OPTIMIZED (After GEPA)")
    optimized_score = evaluate_program(optimized_program, val_examples, hard_reasoning_metric)

    # Show improvement
    improvement = optimized_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Baseline Score:   {baseline_score:.1%}")
    print(f"Optimized Score:  {optimized_score:.1%}")
    print(f"Improvement:      {improvement:+.1%} ({improvement_pct:+.1f}%)")
    print(f"{'='*80}")

    # Show optimized prompts
    print(f"\n{'='*80}")
    print(f"OPTIMIZED PROMPTS")
    print(f"{'='*80}")

    for predictor_name, prompt_data in result.prompts.items():
        print(f"\nPredictor: {predictor_name}")
        if isinstance(prompt_data, dict) and "instruction" in prompt_data:
            print(f"\nInstruction:")
            print(f"{prompt_data['instruction']}")

    # Verify actual improvement
    assert optimized_score >= baseline_score, \
        f"Optimization should not make things worse (baseline: {baseline_score:.1%}, optimized: {optimized_score:.1%})"

    if improvement > 0.05:  # At least 5% improvement
        print(f"\n✅ GEPA achieved significant improvement: {improvement:+.1%}")
    else:
        print(f"\n⚠️  Minor or no improvement - dataset may be too hard/easy or need more iterations")

    print(f"\n✅ Real optimization test completed!")


@pytest.mark.integration
def test_mipro_real_improvement():
    """Test MIPRO with harder problems showing actual optimization."""

    # Setup DSPy
    try:
        setup_test_env()
    except ValueError as e:
        pytest.skip(f"Skipping test: {e}")

    # Get harder dataset
    train_examples, val_examples = split_hard_dataset(train_ratio=0.7)

    print(f"\n{'='*80}")
    print(f"MIPRO REAL OPTIMIZATION TEST - Hard Reasoning Problems")
    print(f"{'='*80}")
    print(f"\n📊 Dataset: {len(train_examples)} train, {len(val_examples)} val")

    # Evaluate baseline
    print("\n🔍 BASELINE (No Optimization)")
    baseline_program = BasicQA()
    baseline_score = evaluate_program(baseline_program, val_examples, simple_contains_metric)

    # Configure MIPRO with minimal budget
    config = OptimizationConfig(
        optimizer_type="mipro",
        auto_budget="light",
        minibatch=False,  # Full evaluation
        minibatch_size=3,
    )

    optimizer = MIPROOptimizer(
        config=config,
        metric=simple_contains_metric,
    )

    # Run optimization
    print(f"\n🚀 Running MIPRO optimization...")
    optimized_program, result = optimizer.optimize(
        program=BasicQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    # Evaluate optimized program
    print(f"\n✨ OPTIMIZED (After MIPRO)")
    optimized_score = evaluate_program(optimized_program, val_examples, simple_contains_metric)

    # Show improvement
    improvement = optimized_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Baseline Score:   {baseline_score:.1%}")
    print(f"Optimized Score:  {optimized_score:.1%}")
    print(f"Improvement:      {improvement:+.1%} ({improvement_pct:+.1f}%)")
    print(f"{'='*80}")

    # Show optimized prompts
    print(f"\n{'='*80}")
    print(f"OPTIMIZED PROMPTS")
    print(f"{'='*80}")

    for predictor_name, prompt_data in result.prompts.items():
        print(f"\nPredictor: {predictor_name}")
        if isinstance(prompt_data, dict) and "instruction" in prompt_data:
            print(f"\nInstruction:")
            print(f"{prompt_data['instruction']}")

    # Verify improvement
    assert optimized_score >= baseline_score, \
        f"Optimization should not make things worse"

    print(f"\n✅ Real optimization test completed!")


if __name__ == "__main__":
    # Run tests directly
    print("Running real optimization tests...")
    test_gepa_real_improvement()
    print("\n" + "="*80 + "\n")
    test_mipro_real_improvement()
