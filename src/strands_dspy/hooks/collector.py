"""Training example collection hook for Strands agents."""

import logging
from typing import Any, Dict

from strands.hooks import AfterInvocationEvent, BeforeInvocationEvent, MessageAddedEvent, HookProvider, HookRegistry
from strands.session import SessionManager

from strands_dspy.storage.session_storage import DSPySessionStorage
from strands_dspy.types import InputExtractor, OutputExtractor, SuccessCriteria

logger = logging.getLogger(__name__)


class DSPyTrainingCollector(HookProvider):
    """Collects training examples from successful Strands agent runs.

    This hook monitors agent invocations and stores successful runs as training
    examples for DSPy optimization. It integrates with Strands' lifecycle hooks
    to capture conversation context and results.

    Example:
        ```python
        from strands import Agent
        from strands.session import FileSessionManager
        from strands_dspy import DSPyTrainingCollector

        def success_criteria(result):
            return result.stop_reason == "end_turn", 1.0

        def extract_inputs(messages):
            return {"question": messages[0]["content"][0]["text"]}

        def extract_outputs(result):
            return {"answer": result.message["content"][0]["text"]}

        collector = DSPyTrainingCollector(
            session_manager=FileSessionManager(session_id="user_123"),
            success_criteria=success_criteria,
            input_extractor=extract_inputs,
            output_extractor=extract_outputs
        )

        agent = Agent(
            model="claude-3-5-sonnet-20241022",
            hooks=[collector]
        )
        ```
    """

    def __init__(
        self,
        session_manager: SessionManager,
        success_criteria: SuccessCriteria,
        input_extractor: InputExtractor,
        output_extractor: OutputExtractor,
        collection_agent_id: str = "dspy_training",
        session_id: str = "default",
    ):
        """Initialize the training collector hook.

        Args:
            session_manager: SessionManager instance for storing examples
            success_criteria: Function that determines if a run was successful
                Returns (is_success: bool, score: float)
            input_extractor: Function that extracts structured inputs from messages
                Returns dict of input fields
            output_extractor: Function that extracts structured outputs from result
                Returns dict of output fields
            collection_agent_id: Agent ID to use for storing training data
                (default: "dspy_training")
            session_id: Session ID for storage isolation
                (default: "default")
        """
        self.storage = DSPySessionStorage(session_manager)
        self.session_id = session_id
        self.success_criteria = success_criteria
        self.input_extractor = input_extractor
        self.output_extractor = output_extractor
        self.collection_agent_id = collection_agent_id

        # Track current invocations
        self._current_invocations: Dict[int, Dict[str, Any]] = {}

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register hooks with the Strands agent.

        This is called automatically when the hook provider is added to an agent.

        Args:
            registry: Hook registry to register event handlers with
        """
        registry.on(BeforeInvocationEvent, self.on_before_invocation)
        registry.on(MessageAddedEvent, self.on_message_added)
        registry.on(AfterInvocationEvent, self.on_after_invocation)

    async def on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Capture invocation start and initialize tracking.

        Args:
            event: Event containing agent and invocation context
        """
        invocation_id = id(event)
        self._current_invocations[invocation_id] = {
            "agent_id": event.agent.agent_id,
            "messages": [],
            "session_id": getattr(event.agent.session_manager, "session_id", self.session_id),
        }
        logger.debug(f"Started tracking invocation {invocation_id}")

    async def on_message_added(self, event: MessageAddedEvent) -> None:
        """Track messages during the conversation.

        Args:
            event: Event containing the message that was added
        """
        invocation_id = id(event)
        if invocation_id in self._current_invocations:
            self._current_invocations[invocation_id]["messages"].append(event.message)
            logger.debug(f"Captured message for invocation {invocation_id}")

    async def on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Evaluate and store successful runs as training examples.

        Args:
            event: Event containing the completed agent invocation
        """
        invocation_id = id(event)
        invocation_data = self._current_invocations.pop(invocation_id, None)

        if not invocation_data:
            logger.warning(f"No invocation data found for {invocation_id}")
            return

        try:
            # Get the agent result
            # Note: We need to access the result from the event's agent
            # In practice, the result is in event.agent.last_result or similar
            # For now, we'll assume it's accessible via the agent
            agent = event.agent
            if not hasattr(agent, "last_result") or agent.last_result is None:
                logger.debug("No result available for this invocation")
                return

            result = agent.last_result

            # Check success criteria
            is_success, score = self.success_criteria(result)

            if not is_success:
                logger.debug(f"Invocation {invocation_id} did not meet success criteria")
                return

            # Extract structured inputs/outputs
            inputs = self.input_extractor(invocation_data["messages"])
            outputs = self.output_extractor(result)

            # Store the training example (storage is synchronous)
            self.storage.store_training_example(
                session_id=invocation_data["session_id"],
                agent_id=self.collection_agent_id,
                inputs=inputs,
                outputs=outputs,
                score=score,
                metadata={
                    "source_agent_id": invocation_data["agent_id"],
                    "message_count": len(invocation_data["messages"]),
                    "invocation_id": str(invocation_id),
                },
            )

            logger.info(
                f"Stored training example with score {score} "
                f"(session={invocation_data['session_id']}, "
                f"agent={self.collection_agent_id})"
            )

        except Exception as e:
            logger.error(f"Failed to store training example for invocation {invocation_id}: {e}")
            logger.exception(e)
