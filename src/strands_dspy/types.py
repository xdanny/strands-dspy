"""Type definitions for strands-dspy."""

from typing import Any, Callable, Dict, Literal, Optional

import dspy
from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """Training example extracted from agent execution."""

    inputs: Dict[str, Any] = Field(description="Input fields for the example")
    outputs: Dict[str, Any] = Field(description="Output fields for the example")
    score: float = Field(description="Success score for this example", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about this example"
    )
    timestamp: str = Field(description="ISO timestamp when example was created")

    class Config:
        arbitrary_types_allowed = True


class OptimizationConfig(BaseModel):
    """Configuration for DSPy optimization."""

    optimizer_type: Literal["mipro", "gepa"] = Field(
        description="Which DSPy optimizer to use"
    )
    num_candidates: Optional[int] = Field(
        default=10, description="Number of instruction candidates to generate (ignored if auto_budget is set)", ge=1
    )
    num_trials: Optional[int] = Field(
        default=30, description="Number of optimization trials to run (ignored if auto_budget is set)", ge=1
    )
    auto_budget: Optional[Literal["light", "medium", "heavy"]] = Field(
        default=None, description="Preset optimization budget configuration (light/medium/heavy, or None for manual control)"
    )
    minibatch: bool = Field(default=True, description="Whether to use minibatch evaluation")
    minibatch_size: int = Field(
        default=25, description="Size of evaluation minibatches", ge=1
    )
    track_stats: bool = Field(
        default=True, description="Whether to track detailed optimization statistics"
    )
    max_bootstrapped_demos: Optional[int] = Field(
        default=3, description="Maximum number of bootstrapped demonstrations (MIPRO only)", ge=0
    )
    max_labeled_demos: Optional[int] = Field(
        default=3, description="Maximum number of labeled demonstrations (MIPRO only)", ge=0
    )

    class Config:
        arbitrary_types_allowed = True


class OptimizationResult(BaseModel):
    """Results from a DSPy optimization run."""

    optimizer: Literal["mipro", "gepa"] = Field(description="Which optimizer was used")
    timestamp: str = Field(description="ISO timestamp when optimization completed")
    train_size: int = Field(description="Number of training examples used", ge=0)
    val_size: int = Field(description="Number of validation examples used", ge=0)
    best_score: Optional[float] = Field(
        default=None, description="Best validation score achieved"
    )
    config: OptimizationConfig = Field(description="Configuration used for optimization")
    prompts: Dict[str, Any] = Field(description="Optimized prompts and instructions")
    detailed_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed optimization results (GEPA only)"
    )

    class Config:
        arbitrary_types_allowed = True


# Type aliases for callback functions
SuccessCriteria = Callable[[Any], tuple[bool, float]]
"""Function that evaluates if an agent result was successful.
Args:
    agent_result: The result from an agent invocation
Returns:
    Tuple of (is_success: bool, score: float)
"""

InputExtractor = Callable[[list], Dict[str, Any]]
"""Function that extracts structured inputs from agent messages.
Args:
    messages: List of conversation messages
Returns:
    Dictionary of input fields
"""

OutputExtractor = Callable[[Any], Dict[str, Any]]
"""Function that extracts structured outputs from agent result.
Args:
    agent_result: The result from an agent invocation
Returns:
    Dictionary of output fields
"""

MetricFunction = Callable[[dspy.Example, dspy.Prediction, Any], float | dspy.Prediction]
"""DSPy metric function for evaluating predictions.
Args:
    example: Ground truth example from dataset
    prediction: Predicted output from DSPy program
    trace: Optional execution trace
Returns:
    Score (float) for MIPRO, or dspy.Prediction(score, feedback) for GEPA
"""

ProgramFactory = Callable[[], dspy.Module]
"""Factory function that creates a DSPy Module/Program.
Returns:
    Fresh instance of a DSPy Module
"""
