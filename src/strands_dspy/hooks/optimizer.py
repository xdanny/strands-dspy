"""DSPy optimization hook for automatic prompt optimization."""

import logging

import dspy
from strands.hooks import AfterInvocationEvent, HookProvider, HookRegistry
from strands.session import SessionManager

from strands_dspy.optimizers.gepa import GEPAOptimizer
from strands_dspy.optimizers.mipro import MIPROOptimizer
from strands_dspy.storage.session_storage import DSPySessionStorage
from strands_dspy.types import MetricFunction, OptimizationConfig, ProgramFactory, TrainingExample

logger = logging.getLogger(__name__)


class DSPyOptimizationHook(HookProvider):
    """Automatically triggers DSPy optimization when threshold is met.

    This hook monitors the number of collected training examples and triggers
    MIPRO or GEPA optimization when the threshold is reached. It handles the
    full optimization workflow:
    1. Retrieve training examples from SessionManager
    2. Convert to DSPy format
    3. Run optimization (MIPRO or GEPA)
    4. Store optimized prompts back to SessionManager

    Example:
        ```python
        from strands import Agent
        from strands.session import FileSessionManager
        from strands_dspy import DSPyOptimizationHook, OptimizationConfig
        import dspy

        def metric(example, prediction, trace=None):
            score = 1.0 if example.answer in prediction.answer else 0.0
            return dspy.Prediction(score=score, feedback="Match" if score else "No match")

        def create_program():
            class QA(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.cot = dspy.ChainOfThought("question -> answer")

                def forward(self, question):
                    return self.cot(question=question)

            return QA()

        optimizer = DSPyOptimizationHook(
            session_manager=FileSessionManager(session_id="user_123"),
            config=OptimizationConfig(optimizer_type="gepa"),
            metric=metric,
            program_factory=create_program,
            example_threshold=50
        )

        agent = Agent(
            model="claude-3-5-sonnet-20241022",
            hooks=[optimizer]
        )
        ```
    """

    def __init__(
        self,
        session_manager: SessionManager,
        config: OptimizationConfig,
        metric: MetricFunction,
        program_factory: ProgramFactory,
        example_threshold: int = 50,
        collection_agent_id: str = "dspy_training",
        session_id: str = "default",
        validation_split: float = 0.2,
    ):
        """Initialize the optimization hook.

        Args:
            session_manager: SessionManager instance for data persistence
            config: Optimization configuration
            metric: DSPy metric function
                For MIPRO: Return float/bool
                For GEPA: Return dspy.Prediction(score, feedback)
            program_factory: Factory function creating DSPy program instances
            example_threshold: Number of examples before triggering optimization
                (default: 50)
            collection_agent_id: Agent ID where training data is stored
                (default: "dspy_training")
            session_id: Session ID for storage isolation
                (default: "default")
            validation_split: Fraction of examples to use for validation
                (default: 0.2 = 20%)
        """
        self.storage = DSPySessionStorage(session_manager)
        self.session_id = session_id
        self.config = config
        self.metric = metric
        self.program_factory = program_factory
        self.example_threshold = example_threshold
        self.collection_agent_id = collection_agent_id
        self.validation_split = validation_split

        self._optimization_running = False
        self._last_example_count = 0

        logger.info(
            f"Initialized DSPy optimization hook: optimizer={config.optimizer_type}, "
            f"threshold={example_threshold}, agent={collection_agent_id}"
        )

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register hooks with the Strands agent.

        This is called automatically when the hook provider is added to an agent.

        Args:
            registry: Hook registry to register event handlers with
        """
        registry.on(AfterInvocationEvent, self.check_optimization_trigger)

    async def check_optimization_trigger(self, event: AfterInvocationEvent) -> None:
        """Check if we should trigger optimization.

        Args:
            event: Event containing completed invocation information
        """
        # Skip if optimization is already running
        if self._optimization_running:
            logger.debug("Optimization already running, skipping trigger check")
            return

        try:
            # Get session ID from event if available
            session_id = getattr(event.agent.session_manager, "session_id", self.session_id)

            # Get training example count (storage is synchronous)
            examples = self.storage.retrieve_training_examples(
                session_id=session_id,
                agent_id=self.collection_agent_id,
            )

            example_count = len(examples)

            # Check if we've reached the threshold
            if example_count >= self.example_threshold and example_count > self._last_example_count:
                logger.info(
                    f"Optimization threshold reached: {example_count} examples >= {self.example_threshold}"
                )
                await self.run_optimization(session_id, examples)
                self._last_example_count = example_count

        except Exception as e:
            logger.error(f"Failed to check optimization trigger: {e}")
            logger.exception(e)

    async def run_optimization(
        self,
        session_id: str,
        training_examples: list[TrainingExample],
    ) -> None:
        """Execute DSPy optimization.

        Args:
            session_id: Session ID for storing results
            training_examples: List of training examples to use
        """
        self._optimization_running = True
        logger.info(
            f"Starting DSPy {self.config.optimizer_type.upper()} optimization with "
            f"{len(training_examples)} examples"
        )

        try:
            # Convert to DSPy format
            dspy_examples = self._convert_to_dspy_examples(training_examples)

            # Split into train/val
            split_idx = int(len(dspy_examples) * (1 - self.validation_split))
            trainset = dspy_examples[:split_idx]
            valset = dspy_examples[split_idx:] if self.validation_split > 0 else None

            logger.info(
                f"Split examples: {len(trainset)} train, "
                f"{len(valset) if valset else 0} validation"
            )

            # Create fresh program instance
            program = self.program_factory()

            # Select and initialize optimizer
            if self.config.optimizer_type == "mipro":
                optimizer = MIPROOptimizer(
                    config=self.config,
                    metric=self.metric,
                )
            elif self.config.optimizer_type == "gepa":
                optimizer = GEPAOptimizer(
                    config=self.config,
                    metric=self.metric,
                )
            else:
                raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

            # Run optimization (synchronous, DSPy is not async)
            optimized_program, result = optimizer.optimize(
                program=program,
                trainset=trainset,
                valset=valset,
            )

            # Store optimization results (storage is synchronous)
            self.storage.store_optimized_prompts(
                session_id=session_id,
                agent_id=self.collection_agent_id,
                optimization_result=result,
            )

            logger.info(
                f"DSPy optimization completed successfully! "
                f"Best score: {result.best_score}, "
                f"Prompts stored for session={session_id}"
            )

        except Exception as e:
            logger.error(f"DSPy optimization failed: {e}")
            logger.exception(e)

        finally:
            self._optimization_running = False

    def _convert_to_dspy_examples(
        self,
        examples: list[TrainingExample],
    ) -> list[dspy.Example]:
        """Convert TrainingExample objects to DSPy format.

        Args:
            examples: List of training examples to convert

        Returns:
            List of dspy.Example objects
        """
        dspy_examples = []

        for ex in examples:
            try:
                # Merge inputs and outputs
                data = {**ex.inputs, **ex.outputs}

                # Create DSPy example
                dspy_ex = dspy.Example(**data)

                # Mark input fields
                input_keys = list(ex.inputs.keys())
                dspy_ex = dspy_ex.with_inputs(*input_keys)

                dspy_examples.append(dspy_ex)

            except Exception as e:
                logger.warning(f"Failed to convert training example: {e}")
                continue

        logger.debug(f"Converted {len(dspy_examples)} examples to DSPy format")
        return dspy_examples

    async def trigger_manual_optimization(self, session_id: str | None = None) -> bool:
        """Manually trigger optimization regardless of threshold.

        This can be called programmatically to force optimization.

        Args:
            session_id: Session ID to use (default: self.session_id)

        Returns:
            True if optimization was triggered, False if already running
        """
        if self._optimization_running:
            logger.warning("Optimization already running, cannot trigger manual optimization")
            return False

        session_id = session_id or self.session_id

        # Retrieve examples (storage is synchronous)
        examples = self.storage.retrieve_training_examples(
            session_id=session_id,
            agent_id=self.collection_agent_id,
        )

        if not examples:
            logger.warning("No training examples found for manual optimization")
            return False

        logger.info(f"Manually triggering optimization with {len(examples)} examples")
        await self.run_optimization(session_id, examples)
        return True
