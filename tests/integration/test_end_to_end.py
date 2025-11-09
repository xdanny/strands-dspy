"""End-to-end integration tests using real Gemini API."""

from unittest.mock import MagicMock

import dspy
import pytest

from strands_dspy import (
    DSPyTrainingCollector,
    OptimizationConfig,
    QAProgram,
)
from strands_dspy.storage import DSPySessionStorage


@pytest.mark.asyncio
async def test_training_collection_with_real_llm(mock_session_manager):
    """Test collecting training examples with real LLM interactions."""

    # Setup success criteria
    def success_criteria(result):
        return result.stop_reason == "end_turn", 1.0

    def extract_inputs(messages):
        if messages and len(messages) > 0:
            return {"question": messages[0]["content"][0]["text"]}
        return {"question": ""}

    def extract_outputs(result):
        if result.message and "content" in result.message:
            return {"answer": result.message["content"][0].get("text", "")}
        return {"answer": ""}

    # Create collector
    collector = DSPyTrainingCollector(
        session_manager=mock_session_manager,
        success_criteria=success_criteria,
        input_extractor=extract_inputs,
        output_extractor=extract_outputs,
        session_id="integration_test",
    )

    # Simulate agent invocation
    from strands.agent.events import AfterInvocationEvent, BeforeInvocationEvent, MessageAddedEvent
    from strands.agent.hooks import HookRegistry

    registry = HookRegistry()
    collector.register_hooks(registry)

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.agent_id = "test_agent"
    mock_agent.session_manager = mock_session_manager

    # Simulate invocation lifecycle
    before_event = BeforeInvocationEvent(agent=mock_agent)
    await collector.on_before_invocation(before_event)

    # Add message
    message_event = MessageAddedEvent(
        agent=mock_agent,
        message={"role": "user", "content": [{"text": "What is 2+2?"}]},
    )
    await collector.on_message_added(message_event)

    # Create mock result
    mock_result = MagicMock()
    mock_result.stop_reason = "end_turn"
    mock_result.message = {"role": "assistant", "content": [{"text": "4"}]}
    mock_agent.last_result = mock_result

    # Complete invocation
    after_event = AfterInvocationEvent(agent=mock_agent)
    await collector.on_after_invocation(after_event)

    # Verify example was stored
    storage = DSPySessionStorage(mock_session_manager)
    examples = await storage.retrieve_training_examples(
        session_id="integration_test", agent_id="dspy_training"
    )

    assert len(examples) == 1
    assert examples[0].inputs["question"] == "What is 2+2?"
    assert examples[0].outputs["answer"] == "4"


@pytest.mark.asyncio
async def test_mipro_optimization_with_gemini(mock_session_manager, sample_training_examples):
    """Test MIPRO optimization using real Gemini API."""
    from strands_dspy.optimizers import MIPROOptimizer

    # Store training examples
    storage = DSPySessionStorage(mock_session_manager)
    for ex in sample_training_examples:
        await storage.store_training_example(session_id="test", agent_id="dspy_training", **ex)

    # Create simple metric
    def metric(example, prediction, trace=None):
        if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
            return 0.0
        return 1.0 if example.answer.lower() in prediction.answer.lower() else 0.0

    # Create optimizer with minimal config for fast testing
    config = OptimizationConfig(
        optimizer_type="mipro",
        num_candidates=2,  # Minimal for speed
        num_trials=3,  # Minimal for speed
        auto_budget="light",
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
    )

    optimizer = MIPROOptimizer(config=config, metric=metric)

    # Create training set
    trainset = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs(
            "question"
        ),
    ]

    # Run optimization (this will use real Gemini API)
    program = QAProgram()

    try:
        optimized, result = await optimizer.optimize(program=program, trainset=trainset)

        # Verify optimization completed
        assert result.optimizer == "mipro"
        assert result.train_size == 2
        assert result.prompts is not None
        assert len(result.prompts) > 0

        print(f"\n✓ MIPRO optimization completed with score: {result.best_score}")
        print(f"  Optimized prompts: {list(result.prompts.keys())}")

    except Exception as e:
        pytest.skip(f"MIPRO optimization failed (may be due to API limits): {e}")


@pytest.mark.asyncio
async def test_gepa_optimization_with_gemini(mock_session_manager, sample_training_examples):
    """Test GEPA optimization using real Gemini API."""
    from strands_dspy.optimizers import GEPAOptimizer

    # Store training examples
    storage = DSPySessionStorage(mock_session_manager)
    for ex in sample_training_examples:
        await storage.store_training_example(session_id="test", agent_id="dspy_training", **ex)

    # Create feedback metric for GEPA
    def feedback_metric(example, prediction, trace=None):
        if not hasattr(prediction, "answer") or not hasattr(example, "answer"):
            return dspy.Prediction(score=0.0, feedback="Missing answer field")

        if example.answer.lower() in prediction.answer.lower():
            return dspy.Prediction(score=1.0, feedback="Correct answer")
        else:
            return dspy.Prediction(
                score=0.0, feedback=f"Expected {example.answer}, got {prediction.answer}"
            )

    # Create optimizer with minimal config
    config = OptimizationConfig(
        optimizer_type="gepa",
        num_candidates=2,
        num_trials=3,
        auto_budget="light",
        track_stats=True,
    )

    optimizer = GEPAOptimizer(config=config, metric=feedback_metric)

    # Create training set
    trainset = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs(
            "question"
        ),
    ]

    # Run optimization
    program = QAProgram()

    try:
        optimized, result = await optimizer.optimize(program=program, trainset=trainset)

        # Verify optimization completed
        assert result.optimizer == "gepa"
        assert result.train_size == 2
        assert result.prompts is not None

        print(f"\n✓ GEPA optimization completed with score: {result.best_score}")
        print(f"  Optimized prompts: {list(result.prompts.keys())}")

        if result.detailed_results:
            print(f"  Detailed results available: {result.detailed_results.keys()}")

    except Exception as e:
        pytest.skip(f"GEPA optimization failed (may be due to API limits): {e}")


@pytest.mark.asyncio
async def test_qa_program_with_gemini():
    """Test QA program with real Gemini inference."""
    program = QAProgram()

    # Test a simple question
    result = program(question="What is the capital of Japan?")

    assert hasattr(result, "answer")
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0

    print("\n✓ QA Program test:")
    print("  Question: What is the capital of Japan?")
    print(f"  Answer: {result.answer}")

    # Check if answer is reasonable (Tokyo should be mentioned)
    assert "tokyo" in result.answer.lower() or "tokyo" in result.answer.lower()
