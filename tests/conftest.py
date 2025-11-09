"""Pytest configuration and fixtures for strands-dspy tests."""

import pytest
import dspy
from unittest.mock import AsyncMock, MagicMock
from strands.session import SessionManager
from strands.types.session import SessionAgent, SessionMessage

from tests.test_config import setup_test_env


# Setup test environment once for all tests
@pytest.fixture(scope="session", autouse=True)
def setup_dspy():
    """Setup DSPy with Gemini for all tests."""
    try:
        setup_test_env()
        yield
    except ValueError as e:
        pytest.skip(f"Skipping tests: {e}")


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager for testing."""
    manager = MagicMock(spec=SessionManager)
    manager.session_id = "test_session"

    # Mock storage
    manager._messages = {}

    def create_message(session_id, agent_id, session_message, **kwargs):
        key = (session_id, agent_id)
        if key not in manager._messages:
            manager._messages[key] = []
        manager._messages[key].append(session_message)
        return len(manager._messages[key]) - 1

    def list_messages(session_id, agent_id, limit=None, offset=0, **kwargs):
        key = (session_id, agent_id)
        messages = manager._messages.get(key, [])
        return messages

    manager.create_message = create_message
    manager.list_messages = list_messages

    return manager


@pytest.fixture
def sample_training_examples():
    """Sample training examples for testing."""
    return [
        {
            "inputs": {"question": "What is 2+2?"},
            "outputs": {"answer": "4"},
            "score": 1.0,
            "metadata": {"test": True}
        },
        {
            "inputs": {"question": "What is the capital of France?"},
            "outputs": {"answer": "Paris"},
            "score": 1.0,
            "metadata": {"test": True}
        },
        {
            "inputs": {"question": "Who wrote Hamlet?"},
            "outputs": {"answer": "William Shakespeare"},
            "score": 0.9,
            "metadata": {"test": True}
        },
    ]


@pytest.fixture
def sample_dspy_examples():
    """Sample DSPy examples for testing."""
    return [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        dspy.Example(question="Who wrote Hamlet?", answer="William Shakespeare").with_inputs("question"),
    ]


@pytest.fixture
def simple_metric():
    """Simple metric function for testing."""
    def metric(example, prediction, trace=None):
        if hasattr(prediction, "answer") and hasattr(example, "answer"):
            return 1.0 if example.answer.lower() in prediction.answer.lower() else 0.0
        return 0.0
    return metric


@pytest.fixture
def feedback_metric():
    """Feedback metric for GEPA testing."""
    def metric(example, prediction, trace=None):
        score = 1.0 if example.answer.lower() in prediction.answer.lower() else 0.0
        feedback = "Correct" if score else "Incorrect"
        return dspy.Prediction(score=score, feedback=feedback)
    return metric


@pytest.fixture
def mock_agent_result():
    """Mock agent result for testing."""
    result = MagicMock()
    result.stop_reason = "end_turn"
    result.message = {
        "role": "assistant",
        "content": [{"text": "Test answer"}]
    }
    return result


@pytest.fixture
def sample_messages():
    """Sample message list for testing."""
    return [
        {
            "role": "user",
            "content": [{"text": "Test question"}]
        }
    ]
