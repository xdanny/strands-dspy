"""Run optimization and display optimized prompts in detail."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dspy
from strands_dspy.optimizers.mipro import MIPROOptimizer
from strands_dspy.types import OptimizationConfig
from tests.fixtures import split_golden_dataset, contains_metric
from tests.test_config import setup_test_env


class SimpleQA(dspy.Module):
    """Simple QA program for testing optimization."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate_answer(question=question)
        return dspy.Prediction(answer=result.answer)


def main():
    """Run MIPRO optimization and display optimized prompts."""

    print("=" * 80)
    print("MIPRO Prompt Optimization Example")
    print("=" * 80)

    # Setup DSPy with OpenRouter
    print("\n📡 Setting up DSPy with OpenRouter MiniMax M2...")
    setup_test_env()

    # Load dataset
    print("\n📊 Loading golden dataset...")
    train_examples, val_examples = split_golden_dataset(train_ratio=0.7)
    train_examples = train_examples[:5]  # Small dataset for quick test
    val_examples = val_examples[:2]

    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")

    # Configure optimizer
    config = OptimizationConfig(
        optimizer_type="mipro",
        auto_budget="light",
        minibatch=True,
        minibatch_size=2,
    )

    optimizer = MIPROOptimizer(
        config=config,
        metric=contains_metric,
    )

    # Run optimization
    print("\n🚀 Running MIPRO optimization...")
    print("   This will take a few minutes...")

    optimized_program, result = optimizer.optimize(
        program=SimpleQA(),
        trainset=train_examples,
        valset=val_examples,
    )

    # Display results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print(f"\n✨ Best Score: {result.best_score:.2%}")
    print(f"📅 Timestamp: {result.timestamp}")
    print(f"📈 Training Examples: {result.train_size}")
    print(f"📊 Validation Examples: {result.val_size}")

    # Display optimized prompts
    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPTS")
    print("=" * 80)

    for predictor_name, prompt_data in result.prompts.items():
        print(f"\n{'─' * 80}")
        print(f"Predictor: {predictor_name}")
        print(f"{'─' * 80}")

        if isinstance(prompt_data, dict):
            # Display instruction
            if "instruction" in prompt_data:
                print(f"\n📝 Instruction:")
                print(f"   {prompt_data['instruction']}")

            # Display demonstrations
            if "demos" in prompt_data and prompt_data["demos"]:
                print(f"\n💡 Few-Shot Demonstrations ({len(prompt_data['demos'])} examples):")
                for i, demo in enumerate(prompt_data["demos"][:3], 1):  # Show first 3
                    print(f"\n   Demo {i}:")
                    for key, value in demo.items():
                        print(f"      {key}: {value[:100]}...")
                if len(prompt_data["demos"]) > 3:
                    print(f"   ... and {len(prompt_data['demos']) - 3} more demos")
            else:
                print("\n💡 No few-shot demonstrations")
        else:
            print(f"\n   {prompt_data}")

    # Test the optimized program
    print("\n" + "=" * 80)
    print("TESTING OPTIMIZED PROGRAM")
    print("=" * 80)

    test_questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "Who wrote Romeo and Juliet?"
    ]

    for question in test_questions:
        prediction = optimized_program(question=question)
        print(f"\n❓ Q: {question}")
        print(f"✅ A: {prediction.answer}")

    # Save results to JSON
    output_file = Path(__file__).parent / "optimization_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "best_score": result.best_score,
            "timestamp": result.timestamp,
            "train_size": result.train_size,
            "val_size": result.val_size,
            "optimizer": result.optimizer,
            "prompts": result.prompts,
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
