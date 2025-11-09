"""
Basic Q&A optimization example using strands-dspy.

This example demonstrates:
1. Setting up a Strands agent with DSPy optimization
2. Collecting training examples automatically
3. Triggering MIPRO optimization after 50 examples
4. Using optimized prompts to improve agent performance
"""

import asyncio
import os
import dspy
from dotenv import load_dotenv
from strands import Agent
from strands.session import FileSessionManager

from strands_dspy import (
    DSPyOptimizationHook,
    DSPyTrainingCollector,
    OptimizationConfig,
    QAProgram,
)

# Load environment variables
load_dotenv()


# Sample Q&A dataset for testing
QA_EXAMPLES = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the speed of light?", "answer": "299,792,458 meters per second"},
    {"question": "What is the smallest prime number?", "answer": "2"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
]


def success_criteria(result):
    """Determine if an agent run was successful.

    For this example, we consider any completed run as successful.
    In production, you might check for specific patterns, user feedback, etc.
    """
    if result.stop_reason == "end_turn":
        # Could add more sophisticated checks here
        return True, 1.0
    return False, 0.0


def extract_inputs(messages):
    """Extract structured inputs from conversation messages.

    Args:
        messages: List of conversation messages

    Returns:
        Dict with 'question' field
    """
    if messages and len(messages) > 0:
        first_msg = messages[0]
        if "content" in first_msg and len(first_msg["content"]) > 0:
            content_block = first_msg["content"][0]
            if isinstance(content_block, dict) and "text" in content_block:
                return {"question": content_block["text"]}

    return {"question": ""}


def extract_outputs(result):
    """Extract structured outputs from agent result.

    Args:
        result: Agent invocation result

    Returns:
        Dict with 'answer' field
    """
    if result.message and "content" in result.message:
        content = result.message["content"]
        if len(content) > 0 and isinstance(content[0], dict):
            return {"answer": content[0].get("text", "")}

    return {"answer": ""}


def create_qa_metric():
    """Create a GEPA-compatible metric function with feedback."""

    def qa_metric_with_feedback(example, prediction, trace=None):
        """Evaluate prediction with feedback for GEPA."""
        # Simple substring match for this example
        # In production, use more sophisticated evaluation
        answer_lower = example.answer.lower()
        pred_lower = prediction.answer.lower()

        if answer_lower in pred_lower:
            return dspy.Prediction(
                score=1.0, feedback="Correct answer provided"
            )
        else:
            return dspy.Prediction(
                score=0.0,
                feedback=f"Expected answer containing '{example.answer}', but got '{prediction.answer}'"
            )

    return qa_metric_with_feedback


def create_program():
    """Factory function to create a fresh DSPy QA program."""
    return QAProgram()


async def main():
    """Run the basic Q&A optimization example."""
    print("=" * 60)
    print("strands-dspy: Basic Q&A Optimization Example")
    print("=" * 60)

    # Setup session manager
    session_manager = FileSessionManager(session_id="qa_example")
    print("\n✓ Created session manager (session_id='qa_example')")

    # Create training collector hook
    collector = DSPyTrainingCollector(
        session_manager=session_manager,
        success_criteria=success_criteria,
        input_extractor=extract_inputs,
        output_extractor=extract_outputs,
    )
    print("✓ Created training collector hook")

    # Create optimization hook with GEPA
    optimizer = DSPyOptimizationHook(
        session_manager=session_manager,
        config=OptimizationConfig(
            optimizer_type="gepa",  # Using GEPA for rich feedback
            num_candidates=5,  # Small number for quick demo
            num_trials=10,  # Small number for quick demo
            auto_budget="light",
        ),
        metric=create_qa_metric(),
        program_factory=create_program,
        example_threshold=10,  # Low threshold for demo (normally 50+)
    )
    print("✓ Created GEPA optimization hook (threshold=10 examples)")

    # Create agent with hooks
    agent = Agent(
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a helpful Q&A assistant. Answer questions accurately and concisely.",
        session_manager=session_manager,
        hooks=[collector, optimizer],
    )
    print("✓ Created Strands agent with optimization hooks\n")

    # Run questions to collect training data
    print("Running Q&A examples to collect training data...")
    print("-" * 60)

    for i, example in enumerate(QA_EXAMPLES, 1):
        question = example["question"]
        expected_answer = example["answer"]

        print(f"\n[{i}/{len(QA_EXAMPLES)}] Question: {question}")

        # Invoke agent
        result = await agent.invoke_async(question)

        # Extract answer
        answer = ""
        if result.message and "content" in result.message:
            content = result.message["content"]
            if len(content) > 0 and isinstance(content[0], dict):
                answer = content[0].get("text", "")

        print(f"         Answer: {answer[:100]}...")
        print(f"         Expected: {expected_answer}")

        # Note: Training collection and optimization happen automatically via hooks!

    print("\n" + "=" * 60)
    print("Training data collection complete!")
    print("=" * 60)
    print("\nOptimization will trigger automatically when threshold is reached.")
    print("Check the session storage for optimized prompts.")
    print(f"\nSession data stored in: sessions/qa_example/")


if __name__ == "__main__":
    # Configure DSPy with Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please copy .env.example to .env and add your Gemini API key.")
        exit(1)

    # Configure DSPy with Gemini 2.5 Flash Lite
    lm = dspy.LM(
        model="google/gemini-2.0-flash-lite",
        api_key=api_key,
        temperature=0.7,
    )
    dspy.configure(lm=lm)

    print("✓ Configured DSPy with Gemini 2.5 Flash Lite\n")

    asyncio.run(main())
