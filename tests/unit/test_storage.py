"""Tests for DSPySessionStorage."""

import pytest

from strands_dspy.storage import DSPySessionStorage
from strands_dspy.types import OptimizationConfig, OptimizationResult


def test_store_training_example(mock_session_manager):
    """Test storing a training example."""
    storage = DSPySessionStorage(mock_session_manager)

    storage.store_training_example(
        session_id="test",
        agent_id="test_agent",
        inputs={"question": "What is 2+2?"},
        outputs={"answer": "4"},
        score=1.0,
        metadata={"test": True},
    )

    # Verify message was created
    key = ("test", "test_agent")
    assert key in mock_session_manager._messages
    assert len(mock_session_manager._messages[key]) >= 1


def test_retrieve_training_examples(mock_session_manager, sample_training_examples):
    """Test retrieving training examples."""
    storage = DSPySessionStorage(mock_session_manager)

    # Store examples
    for ex in sample_training_examples:
        storage.store_training_example(session_id="test", agent_id="test_agent", **ex)

    # Retrieve examples
    examples = storage.retrieve_training_examples(session_id="test", agent_id="test_agent")

    assert len(examples) == 3
    assert examples[0].inputs["question"] == "What is 2+2?"
    assert examples[0].outputs["answer"] == "4"
    assert examples[0].score == 1.0


def test_retrieve_with_min_score(mock_session_manager, sample_training_examples):
    """Test filtering examples by minimum score."""
    storage = DSPySessionStorage(mock_session_manager)

    # Store examples
    for ex in sample_training_examples:
        storage.store_training_example(session_id="test", agent_id="test_agent", **ex)

    # Retrieve with min_score filter
    examples = storage.retrieve_training_examples(
        session_id="test", agent_id="test_agent", min_score=1.0
    )

    # Should only get examples with score >= 1.0
    assert len(examples) == 2
    assert all(ex.score >= 1.0 for ex in examples)


def test_retrieve_with_limit(mock_session_manager, sample_training_examples):
    """Test limiting number of retrieved examples."""
    storage = DSPySessionStorage(mock_session_manager)

    # Store examples
    for ex in sample_training_examples:
        storage.store_training_example(session_id="test", agent_id="test_agent", **ex)

    # Retrieve with limit
    examples = storage.retrieve_training_examples(session_id="test", agent_id="test_agent", limit=2)

    assert len(examples) == 2


def test_store_optimized_prompts(mock_session_manager):
    """Test storing optimization results."""
    storage = DSPySessionStorage(mock_session_manager)

    result = OptimizationResult(
        optimizer="mipro",
        timestamp="2024-01-01T00:00:00",
        train_size=10,
        val_size=2,
        best_score=0.95,
        config=OptimizationConfig(optimizer_type="mipro"),
        prompts={"predictor": {"instruction": "Test instruction"}},
    )

    storage.store_optimized_prompts(
        session_id="test", agent_id="test_agent", optimization_result=result
    )

    # Verify message was created
    key = ("test", "test_agent")
    assert key in mock_session_manager._messages
    assert len(mock_session_manager._messages[key]) >= 1


def test_retrieve_latest_optimization(mock_session_manager):
    """Test retrieving the latest optimization result."""
    storage = DSPySessionStorage(mock_session_manager)

    # Store multiple optimizations
    for i in range(3):
        result = OptimizationResult(
            optimizer="mipro",
            timestamp=f"2024-01-0{i+1}T00:00:00",
            train_size=10 + i,
            val_size=2,
            best_score=0.9 + (i * 0.01),
            config=OptimizationConfig(optimizer_type="mipro"),
            prompts={"predictor": {"instruction": f"Instruction {i}"}},
        )
        storage.store_optimized_prompts(
            session_id="test", agent_id="test_agent", optimization_result=result
        )

    # Retrieve latest
    latest = storage.retrieve_latest_optimization(session_id="test", agent_id="test_agent")

    assert latest is not None
    assert latest.train_size == 12  # Last one stored


def test_get_training_stats(mock_session_manager, sample_training_examples):
    """Test getting training statistics."""
    storage = DSPySessionStorage(mock_session_manager)

    # Store examples
    for ex in sample_training_examples:
        storage.store_training_example(session_id="test", agent_id="test_agent", **ex)

    # Get stats
    stats = storage.get_training_stats(session_id="test", agent_id="test_agent")

    assert stats["count"] == 3
    assert stats["avg_score"] == pytest.approx((1.0 + 1.0 + 0.9) / 3)
    assert stats["max_score"] == 1.0
    assert stats["min_score"] == 0.9
