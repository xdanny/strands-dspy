"""SessionManager-based storage for DSPy training examples and optimized prompts."""

import json
import logging
from datetime import datetime
from typing import Any

from strands.session import SessionManager
from strands.types.content import Message
from strands.types.session import SessionMessage

from strands_dspy.types import OptimizationResult, TrainingExample

logger = logging.getLogger(__name__)


class DSPySessionStorage:
    """Stores DSPy training examples and optimizations using Strands SessionManager.

    This class provides utilities for storing and retrieving:
    - Training examples collected from successful agent runs
    - Optimized prompts and instructions from DSPy optimization
    - Optimization metadata and statistics

    All data is stored using SessionManager's native message and agent storage,
    ensuring compatibility with Strands' session persistence mechanisms.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize storage with a SessionManager instance.

        Args:
            session_manager: Strands SessionManager for data persistence
        """
        self.session_manager = session_manager

    def store_training_example(
        self,
        session_id: str,
        agent_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a training example as a session message.

        Training examples are stored as assistant messages with special JSON content
        that marks them as DSPy training data.

        Args:
            session_id: Session identifier for storage isolation
            agent_id: Agent identifier (typically "dspy_training")
            inputs: Input fields for the training example
            outputs: Output fields for the training example
            score: Success score (0.0 to 1.0)
            metadata: Optional metadata about the example
        """
        example = TrainingExample(
            inputs=inputs,
            outputs=outputs,
            score=score,
            metadata=metadata or {},
            timestamp=datetime.utcnow().isoformat(),
        )

        # Create message content
        message_content = Message(
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "dspy_training_example": True,
                            "data": example.model_dump(),
                        }
                    ),
                }
            ],
        )

        # Get next message ID by checking existing messages
        existing_messages = self.session_manager.list_messages(
            session_id=session_id, agent_id=agent_id
        )
        next_message_id = len(existing_messages) if existing_messages else 0

        # Create session message
        session_message = SessionMessage(
            message=message_content,
            message_id=next_message_id,
        )

        self.session_manager.create_message(
            session_id=session_id,
            agent_id=agent_id,
            session_message=session_message,
        )

        logger.debug(
            f"Stored training example with score {score} for session={session_id}, agent={agent_id}"
        )

    def retrieve_training_examples(
        self,
        session_id: str,
        agent_id: str,
        min_score: float | None = None,
        limit: int | None = None,
    ) -> list[TrainingExample]:
        """Retrieve training examples from session storage.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier where examples are stored
            min_score: Optional minimum score filter
            limit: Optional maximum number of examples to return

        Returns:
            List of training examples, ordered by storage time
        """
        # List all messages for this agent
        messages = self.session_manager.list_messages(
            session_id=session_id,
            agent_id=agent_id,
        )

        if not messages:
            logger.debug(f"No messages found for session={session_id}, agent={agent_id}")
            return []

        examples = []
        for session_msg in messages:
            msg = session_msg.message  # Extract the Message object
            if msg.get("role") == "assistant":
                for content_block in msg.get("content", []):
                    if isinstance(content_block, dict) and "text" in content_block:
                        try:
                            data = json.loads(content_block["text"])
                            if data.get("dspy_training_example"):
                                example_data = data["data"]
                                example = TrainingExample(**example_data)

                                # Apply filters
                                if min_score is None or example.score >= min_score:
                                    examples.append(example)

                                    # Check limit
                                    if limit and len(examples) >= limit:
                                        return examples

                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning(f"Failed to parse training example: {e}")
                            continue

        logger.debug(
            f"Retrieved {len(examples)} training examples for session={session_id}, agent={agent_id}"
        )
        return examples

    def store_optimized_prompts(
        self,
        session_id: str,
        agent_id: str,
        optimization_result: OptimizationResult,
    ) -> None:
        """Store optimized prompts and metadata from DSPy optimization.

        Optimization results are stored as system messages, making them distinct
        from training examples and easily retrievable.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            optimization_result: Complete optimization result with prompts and metadata
        """
        # Create message content
        message_content = Message(
            role="system",
            content=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "dspy_optimized_prompts": True,
                            "data": optimization_result.model_dump(),
                        }
                    ),
                }
            ],
        )

        # Get next message ID by checking existing messages
        existing_messages = self.session_manager.list_messages(
            session_id=session_id, agent_id=agent_id
        )
        next_message_id = len(existing_messages) if existing_messages else 0

        # Create session message
        session_message = SessionMessage(
            message=message_content,
            message_id=next_message_id,
        )

        self.session_manager.create_message(
            session_id=session_id,
            agent_id=agent_id,
            session_message=session_message,
        )

        logger.info(
            f"Stored optimized prompts ({optimization_result.optimizer}) for session={session_id}, agent={agent_id}"
        )

    def retrieve_latest_optimization(
        self,
        session_id: str,
        agent_id: str,
    ) -> OptimizationResult | None:
        """Retrieve the most recent optimization result.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            Latest optimization result, or None if no optimization found
        """
        messages = self.session_manager.list_messages(
            session_id=session_id,
            agent_id=agent_id,
        )

        if not messages:
            return None

        # Search messages in reverse order (newest first)
        for session_msg in reversed(messages):
            msg = session_msg.message  # Extract the Message object
            if msg.get("role") == "system":
                for content_block in msg.get("content", []):
                    if isinstance(content_block, dict) and "text" in content_block:
                        try:
                            data = json.loads(content_block["text"])
                            if data.get("dspy_optimized_prompts"):
                                return OptimizationResult(**data["data"])
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning(f"Failed to parse optimization result: {e}")
                            continue

        return None

    def retrieve_all_optimizations(
        self,
        session_id: str,
        agent_id: str,
    ) -> list[OptimizationResult]:
        """Retrieve all optimization results in chronological order.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            List of all optimization results
        """
        messages = self.session_manager.list_messages(
            session_id=session_id,
            agent_id=agent_id,
        )

        if not messages:
            return []

        optimizations = []
        for session_msg in messages:
            msg = session_msg.message  # Extract the Message object
            if msg.get("role") == "system":
                for content_block in msg.get("content", []):
                    if isinstance(content_block, dict) and "text" in content_block:
                        try:
                            data = json.loads(content_block["text"])
                            if data.get("dspy_optimized_prompts"):
                                optimizations.append(OptimizationResult(**data["data"]))
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning(f"Failed to parse optimization result: {e}")
                            continue

        return optimizations

    def get_training_stats(
        self,
        session_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """Get statistics about stored training examples.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            Dictionary with statistics (count, avg_score, score_distribution, etc.)
        """
        examples = self.retrieve_training_examples(
            session_id=session_id,
            agent_id=agent_id,
        )

        if not examples:
            return {
                "count": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }

        scores = [ex.score for ex in examples]

        return {
            "count": len(examples),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": {
                "low": len([s for s in scores if s < 0.33]),
                "medium": len([s for s in scores if 0.33 <= s < 0.66]),
                "high": len([s for s in scores if s >= 0.66]),
            },
        }
