"""Blueprint: Question-Answering Agent with DSPy Optimization

This blueprint shows a complete working example of optimizing a Q&A agent
using GEPA optimizer with pre-built helpers.

Usage:
    1. Copy this file to your project
    2. Update the TRAINING_DATA with your examples
    3. Configure your LM (OpenRouter, Gemini, etc.)
    4. Run optimization

Example:
    $ uv run python qa_blueprint.py
"""

import os
from dotenv import load_dotenv
import dspy
from strands import Agent
from strands_dspy import (
    TrainingCollector,
    GEPAOptimizer,
    SessionStorageBackend,
)
from strands_dspy.helpers import (
    extract_first_user_text,
    extract_last_assistant_text,
    end_turn_success,
    contains_match_gepa,
)
from strands_dspy.types import OptimizationConfig


# ============================================================================
# 1. Configure your LM
# ============================================================================

load_dotenv()

# Example: OpenRouter with free tier
lm = dspy.LM(
    model="openrouter/minimax/minimax-m2:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0,
)

# Or use Gemini:
# lm = dspy.LM(
#     model="gemini/gemini-2.5-flash-lite",
#     api_key=os.getenv("GEMINI_API_KEY"),
#     temperature=0.0,
# )

dspy.configure(lm=lm)


# ============================================================================
# 2. Define your training data
# ============================================================================

TRAINING_DATA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the capital of Brazil?", "answer": "Brasília"},
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa"},
    # Add more examples here...
]


# ============================================================================
# 3. Create your agent
# ============================================================================

qa_agent = Agent(
    name="qa-agent",
    instructions="""You are a helpful Q&A assistant.
Answer questions accurately and concisely.""",
    model="gemini/gemini-2.5-flash-lite",  # or your preferred model
)


# ============================================================================
# 4. Set up training collection (using pre-built helpers!)
# ============================================================================

storage = SessionStorageBackend()

collector = TrainingCollector(
    storage=storage,
    # Pre-built extractors - no custom code needed!
    input_extractor=extract_first_user_text,
    output_extractor=extract_last_assistant_text,
    # Pre-built success criteria
    success_criteria=end_turn_success,
)

# Attach collector to agent
collector.attach(qa_agent)


# ============================================================================
# 5. Collect training examples
# ============================================================================

def collect_training_examples():
    """Run agent on training data to collect examples."""
    print(f"\n📊 Collecting training examples from {len(TRAINING_DATA)} Q&A pairs...")

    for i, pair in enumerate(TRAINING_DATA):
        question = pair["question"]
        print(f"  [{i+1}/{len(TRAINING_DATA)}] {question}")

        # Run agent
        qa_agent.invoke(question)

    # Retrieve collected examples
    examples = storage.get_training_examples("session-1", "qa-agent")
    print(f"\n✅ Collected {len(examples)} training examples")

    return examples


# ============================================================================
# 6. Define your DSPy program
# ============================================================================

class QAProgram(dspy.Module):
    """Simple Q&A DSPy program."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)


# ============================================================================
# 7. Create validation/test examples
# ============================================================================

VALIDATION_DATA = [
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "What is the capital of Spain?", "answer": "Madrid"},
]


# ============================================================================
# 8. Run GEPA optimization
# ============================================================================

def run_optimization():
    """Run GEPA optimization using collected examples."""

    # Collect examples first
    training_examples = collect_training_examples()

    # Convert to DSPy format
    trainset = [
        dspy.Example(question=ex.inputs["question"], answer=ex.outputs["answer"]).with_inputs("question")
        for ex in training_examples
    ]

    valset = [
        dspy.Example(question=ex["question"], answer=ex["answer"]).with_inputs("question")
        for ex in VALIDATION_DATA
    ]

    print(f"\n🔧 Starting GEPA optimization...")
    print(f"   Training examples: {len(trainset)}")
    print(f"   Validation examples: {len(valset)}")

    # Configure optimization
    config = OptimizationConfig(
        optimizer_type="gepa",
        auto_budget="light",  # or "medium", "heavy", or None for manual control
        track_stats=True,
    )

    # Create optimizer with pre-built metric!
    optimizer = GEPAOptimizer(
        config=config,
        metric=contains_match_gepa,  # Pre-built GEPA metric - no custom code!
    )

    # Run optimization
    program = QAProgram()
    optimized_program, result = optimizer.optimize(
        program=program,
        trainset=trainset,
        valset=valset,
    )

    print(f"\n✨ Optimization complete!")
    print(f"   Best score: {result.best_score:.2%}")

    # Show optimized prompts
    print(f"\n📝 Optimized prompts:")
    for predictor_name, prompt_data in result.prompts.items():
        print(f"\n{predictor_name}:")
        if "instruction" in prompt_data:
            print(f"  Instruction: {prompt_data['instruction']}")
        if "demos" in prompt_data and prompt_data["demos"]:
            print(f"  Demonstrations: {len(prompt_data['demos'])} examples")

    return optimized_program, result


# ============================================================================
# 9. Test optimized program
# ============================================================================

def test_optimized_program(program):
    """Test the optimized program on new questions."""

    test_questions = [
        "What is the capital of Mexico?",
        "What is the capital of Egypt?",
    ]

    print(f"\n🧪 Testing optimized program...")

    for question in test_questions:
        prediction = program(question=question)
        print(f"\n  Q: {question}")
        print(f"  A: {prediction.answer}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run optimization
    optimized_program, result = run_optimization()

    # Test it
    test_optimized_program(optimized_program)

    print("\n✅ Done!")
