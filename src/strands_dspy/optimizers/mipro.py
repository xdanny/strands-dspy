"""MIPRO optimizer wrapper for strands-dspy."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy
from dspy.teleprompt import MIPROv2

from strands_dspy.types import MetricFunction, OptimizationConfig, OptimizationResult

logger = logging.getLogger(__name__)


class MIPROOptimizer:
    """Wrapper for DSPy's MIPROv2 optimizer.

    MIPRO (Multi-prompt Instruction Proposal Optimizer) automatically generates
    and optimizes instructions and few-shot demonstrations for DSPy programs.

    It works by:
    1. Bootstrapping few-shot examples from successful runs
    2. Generating instruction candidates via an LM
    3. Running Bayesian optimization to find the best combination

    Example:
        ```python
        from strands_dspy.optimizers import MIPROOptimizer
        from strands_dspy.types import OptimizationConfig
        import dspy

        def metric(example, prediction, trace=None):
            return example.answer in prediction.answer

        optimizer = MIPROOptimizer(
            config=OptimizationConfig(
                optimizer_type="mipro",
                num_candidates=10,
                num_trials=30
            ),
            metric=metric
        )

        optimized, result = await optimizer.optimize(
            program=my_program,
            trainset=train_examples,
            valset=val_examples
        )
        ```
    """

    def __init__(
        self,
        config: OptimizationConfig,
        metric: MetricFunction,
        prompt_model: Optional[dspy.LM] = None,
    ):
        """Initialize MIPRO optimizer.

        Args:
            config: Optimization configuration
            metric: Metric function for evaluating predictions
                Should return float or bool (higher is better)
            prompt_model: Optional LM for instruction generation
                If None, uses the default configured LM
        """
        if config.optimizer_type != "mipro":
            raise ValueError(f"Expected optimizer_type='mipro', got '{config.optimizer_type}'")

        self.config = config
        self.metric = metric
        self.prompt_model = prompt_model

        # Initialize MIPROv2 (only __init__ parameters)
        # Note: If auto is set, num_candidates cannot be passed (will be auto-determined)
        init_kwargs = {
            "metric": metric,
            "auto": config.auto_budget,
            "prompt_model": prompt_model,
            "max_bootstrapped_demos": config.max_bootstrapped_demos or 4,
            "max_labeled_demos": config.max_labeled_demos or 4,
            "track_stats": config.track_stats,
        }

        # Only add num_candidates if auto is None (otherwise it conflicts)
        if config.auto_budget is None and config.num_candidates is not None:
            init_kwargs["num_candidates"] = config.num_candidates

        self.optimizer = MIPROv2(**init_kwargs)

        logger.info(
            f"Initialized MIPRO optimizer with {config.num_candidates} candidates, "
            f"budget={config.auto_budget}"
        )

    def optimize(
        self,
        program: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
    ) -> tuple[dspy.Module, OptimizationResult]:
        """Run MIPRO optimization on a DSPy program.

        Args:
            program: DSPy Module to optimize
            trainset: Training examples for optimization
            valset: Optional validation examples for evaluation

        Returns:
            Tuple of (optimized_program, optimization_result)
        """
        logger.info(
            f"Starting MIPRO optimization with {len(trainset)} training examples, "
            f"{len(valset) if valset else 0} validation examples"
        )

        try:
            # Prepare compile parameters
            # Note: If auto is set, num_trials cannot be passed (will be auto-determined)
            compile_kwargs = {
                "student": program,
                "trainset": trainset,
                "valset": valset,
                "minibatch": self.config.minibatch,
                "minibatch_size": self.config.minibatch_size if self.config.minibatch else 35,
                "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                "max_labeled_demos": self.config.max_labeled_demos,
            }

            # Only add num_trials if auto is None (otherwise it conflicts)
            if self.config.auto_budget is None and self.config.num_trials is not None:
                compile_kwargs["num_trials"] = self.config.num_trials

            # Run MIPRO compilation
            optimized = self.optimizer.compile(**compile_kwargs)

            # Extract optimized prompts
            prompts = self._extract_prompts(optimized)

            # Evaluate on validation set if provided
            best_score = None
            if valset:
                best_score = self._evaluate_program(optimized, valset)
                logger.info(f"MIPRO optimization completed with validation score: {best_score}")
            else:
                logger.info("MIPRO optimization completed (no validation set)")

            # Create result
            result = OptimizationResult(
                optimizer="mipro",
                timestamp=datetime.utcnow().isoformat(),
                train_size=len(trainset),
                val_size=len(valset) if valset else 0,
                best_score=best_score,
                config=self.config,
                prompts=prompts,
            )

            return optimized, result

        except Exception as e:
            logger.error(f"MIPRO optimization failed: {e}")
            raise

    def _extract_prompts(self, program: dspy.Module) -> Dict[str, Any]:
        """Extract optimized prompts and demonstrations from a DSPy program.

        Args:
            program: Optimized DSPy Module

        Returns:
            Dictionary mapping predictor names to their prompts and demos
        """
        prompts = {}

        try:
            # Navigate program structure to extract prompts
            for name, module in program.named_predictors():
                prompt_data = {}

                # Extract instruction if available
                if hasattr(module, "signature"):
                    signature = module.signature
                    if hasattr(signature, "instructions"):
                        prompt_data["instruction"] = signature.instructions
                    elif hasattr(signature, "__doc__"):
                        prompt_data["instruction"] = signature.__doc__ or ""

                # Extract demonstrations if available
                if hasattr(module, "demos"):
                    demos = module.demos
                    if demos:
                        prompt_data["demos"] = [
                            {k: str(v) for k, v in demo.items()}
                            for demo in demos
                        ]
                    else:
                        prompt_data["demos"] = []

                prompts[name] = prompt_data

        except Exception as e:
            logger.warning(f"Failed to extract some prompts: {e}")

        return prompts

    def _evaluate_program(
        self,
        program: dspy.Module,
        valset: List[dspy.Example],
    ) -> float:
        """Evaluate program on validation set.

        Args:
            program: DSPy Module to evaluate
            valset: Validation examples

        Returns:
            Average score across validation set
        """
        scores = []

        for example in valset:
            try:
                # Get prediction
                prediction = program(**example.inputs())

                # Evaluate
                score = self.metric(example, prediction)

                # Handle boolean scores
                if isinstance(score, bool):
                    score = 1.0 if score else 0.0

                scores.append(score)

            except Exception as e:
                logger.warning(f"Failed to evaluate example: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0
