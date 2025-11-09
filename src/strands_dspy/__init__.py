"""
strands-dspy: DSPy prompt optimization for Strands agents.

This package provides automatic prompt optimization for Strands agents using
DSPy's MIPRO and GEPA optimizers. It includes:

- Training example collection from successful agent runs
- Automatic optimization trigger based on example threshold
- SessionManager-based storage for all data
- Memory tools for querying optimization results
- Common DSPy program templates

Example:
    ```python
    from strands import Agent
    from strands.session import FileSessionManager
    from strands_dspy import (
        DSPyTrainingCollector,
        DSPyOptimizationHook,
        OptimizationConfig,
    )
    import dspy

    # Setup
    session_manager = FileSessionManager(session_id="user_123")

    collector = DSPyTrainingCollector(
        session_manager=session_manager,
        success_criteria=lambda r: (r.stop_reason == "end_turn", 1.0),
        input_extractor=lambda m: {"question": m[0]["content"][0]["text"]},
        output_extractor=lambda r: {"answer": r.message["content"][0]["text"]},
    )

    optimizer = DSPyOptimizationHook(
        session_manager=session_manager,
        config=OptimizationConfig(optimizer_type="gepa"),
        metric=your_metric,
        program_factory=create_program,
        example_threshold=50,
    )

    # Create agent with hooks
    agent = Agent(
        model="claude-3-5-sonnet-20241022",
        hooks=[collector, optimizer],
    )
    ```
"""

__version__ = "0.1.0"

# Core hooks
from strands_dspy.hooks import DSPyOptimizationHook, DSPyTrainingCollector

# Storage
from strands_dspy.storage import DSPySessionStorage

# Optimizers
from strands_dspy.optimizers import GEPAOptimizer, MIPROOptimizer

# Tools
from strands_dspy.tools import (
    get_training_stats,
    list_training_examples,
    retrieve_dspy_optimizations,
)

# Types
from strands_dspy.types import (
    InputExtractor,
    MetricFunction,
    OptimizationConfig,
    OptimizationResult,
    OutputExtractor,
    ProgramFactory,
    SuccessCriteria,
    TrainingExample,
)

# Module templates (optional, for convenience)
from strands_dspy.modules import (
    QAProgram,
    ClassificationProgram,
    ReasoningProgram,
    RAGProgram,
    QuestionAnswering,
    Classification,
    ReasoningTask,
    SummarizationTask,
    ExtractionTask,
)

__all__ = [
    # Version
    "__version__",
    # Core hooks
    "DSPyTrainingCollector",
    "DSPyOptimizationHook",
    # Storage
    "DSPySessionStorage",
    # Optimizers
    "MIPROOptimizer",
    "GEPAOptimizer",
    # Tools
    "retrieve_dspy_optimizations",
    "list_training_examples",
    "get_training_stats",
    # Types
    "OptimizationConfig",
    "OptimizationResult",
    "TrainingExample",
    "SuccessCriteria",
    "InputExtractor",
    "OutputExtractor",
    "MetricFunction",
    "ProgramFactory",
    # Module templates
    "QAProgram",
    "ClassificationProgram",
    "ReasoningProgram",
    "RAGProgram",
    "QuestionAnswering",
    "Classification",
    "ReasoningTask",
    "SummarizationTask",
    "ExtractionTask",
]
