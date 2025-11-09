"""GEPA optimizer wrapper for strands-dspy."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy
from dspy import GEPA

from strands_dspy.types import MetricFunction, OptimizationConfig, OptimizationResult

logger = logging.getLogger(__name__)


class GEPAOptimizer:
    """Wrapper for DSPy's GEPA optimizer.

    GEPA (Grounded Proposal) uses rich textual feedback and LLM reflection
    to iteratively improve prompts. Unlike MIPRO, GEPA leverages domain-specific
    feedback (not just scores) for more actionable, targeted improvements.

    It works by:
    1. Sampling candidates from a Pareto frontier
    2. Collecting execution traces with feedback
    3. Using LLM reflection to propose improvements
    4. Maintaining diversity via Pareto-based selection

    Example:
        ```python
        from strands_dspy.optimizers import GEPAOptimizer
        from strands_dspy.types import OptimizationConfig
        import dspy

        def feedback_metric(example, prediction, trace=None):
            score = 1.0 if example.answer in prediction.answer else 0.0
            feedback = "Correct" if score else f"Expected {example.answer}"
            return dspy.Prediction(score=score, feedback=feedback)

        optimizer = GEPAOptimizer(
            config=OptimizationConfig(
                optimizer_type="gepa",
                num_candidates=10,
                num_trials=30,
                track_stats=True
            ),
            metric=feedback_metric
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
        reflection_lm: Optional[dspy.LM] = None,
    ):
        """Initialize GEPA optimizer.

        Args:
            config: Optimization configuration
            metric: Feedback metric function
                Must return dspy.Prediction(score=..., feedback=...)
            reflection_lm: Optional LM for reflection
                If None, uses the default configured LM
        """
        if config.optimizer_type != "gepa":
            raise ValueError(f"Expected optimizer_type='gepa', got '{config.optimizer_type}'")

        self.config = config
        self.metric = metric
        self.reflection_lm = reflection_lm

        # Initialize GEPA (using correct API)
        # Note: GEPA uses auto (not auto_budget), max_full_evals (not num_trials),
        # and reflection_minibatch_size (not minibatch_size)
        # IMPORTANT: Only ONE of auto, max_full_evals, max_metric_calls can be set
        # IMPORTANT: reflection_lm is REQUIRED (can't be None)

        # Use provided reflection_lm or default to configured LM
        if reflection_lm is None:
            reflection_lm = dspy.settings.lm

        init_kwargs = {
            "metric": metric,
            "reflection_minibatch_size": config.minibatch_size,
            "track_stats": config.track_stats,
            "track_best_outputs": True,
            "reflection_lm": reflection_lm,
        }

        # Only set ONE budget control parameter
        # GEPA requires exactly one of: auto, max_full_evals, max_metric_calls
        if config.auto_budget:
            init_kwargs["auto"] = config.auto_budget
        elif config.num_trials:
            init_kwargs["max_full_evals"] = config.num_trials
        else:
            # Default to a reasonable budget if neither is set
            init_kwargs["max_full_evals"] = 10

        self.optimizer = GEPA(**init_kwargs)

        logger.info(
            f"Initialized GEPA optimizer with "
            f"auto={config.auto_budget if config.auto_budget else 'None'}, "
            f"max_full_evals={config.num_trials if not config.auto_budget and config.num_trials else 'None'}, "
            f"track_stats={config.track_stats}"
        )

    def optimize(
        self,
        program: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
    ) -> tuple[dspy.Module, OptimizationResult]:
        """Run GEPA optimization on a DSPy program.

        Args:
            program: DSPy Module to optimize
            trainset: Training examples for optimization
            valset: Optional validation examples for evaluation

        Returns:
            Tuple of (optimized_program, optimization_result)
        """
        logger.info(
            f"Starting GEPA optimization with {len(trainset)} training examples, "
            f"{len(valset) if valset else 0} validation examples"
        )

        try:
            # Run GEPA compilation
            optimized = self.optimizer.compile(
                student=program,
                trainset=trainset,
                valset=valset,
            )

            # Extract optimized prompts
            prompts = self._extract_prompts(optimized)

            # Extract detailed results if tracking enabled
            detailed_results = None
            best_score = None

            if self.config.track_stats and hasattr(optimized, "detailed_results"):
                detailed_results = self._extract_detailed_results(optimized.detailed_results)
                best_score = detailed_results.get("best_score")

                logger.info(
                    f"GEPA optimization completed with best score: {best_score}"
                )
            elif valset:
                # Evaluate manually if no stats tracking
                best_score = self._evaluate_program(optimized, valset)
                logger.info(f"GEPA optimization completed with validation score: {best_score}")
            else:
                logger.info("GEPA optimization completed (no validation set)")

            # Create result
            result = OptimizationResult(
                optimizer="gepa",
                timestamp=datetime.utcnow().isoformat(),
                train_size=len(trainset),
                val_size=len(valset) if valset else 0,
                best_score=best_score,
                config=self.config,
                prompts=prompts,
                detailed_results=detailed_results,
            )

            return optimized, result

        except Exception as e:
            logger.error(f"GEPA optimization failed: {e}")
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

    def _extract_detailed_results(self, detailed_results: Any) -> Dict[str, Any]:
        """Extract detailed optimization results from GEPA.

        Args:
            detailed_results: GEPA detailed results object

        Returns:
            Dictionary with scores, best outputs, and other statistics
        """
        try:
            results = {}

            # Extract best outputs if available
            if hasattr(detailed_results, "best_outputs_valset"):
                results["best_outputs_valset"] = detailed_results.best_outputs_valset

            # Extract scores if available
            if hasattr(detailed_results, "highest_score_achieved_per_val_task"):
                scores = detailed_results.highest_score_achieved_per_val_task
                results["scores_per_task"] = scores
                if scores:
                    results["best_score"] = max(scores) if isinstance(scores, list) else scores
                    results["avg_score"] = (
                        sum(scores) / len(scores) if isinstance(scores, list) else scores
                    )

            # Extract Pareto frontier info if available
            if hasattr(detailed_results, "pareto_frontier"):
                results["pareto_frontier_size"] = len(detailed_results.pareto_frontier)

            return results

        except Exception as e:
            logger.warning(f"Failed to extract detailed results: {e}")
            return {}

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
                result = self.metric(example, prediction)

                # Extract score from Prediction or use directly
                if isinstance(result, dspy.Prediction):
                    score = result.score
                elif isinstance(result, bool):
                    score = 1.0 if result else 0.0
                else:
                    score = float(result)

                scores.append(score)

            except Exception as e:
                logger.warning(f"Failed to evaluate example: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0
