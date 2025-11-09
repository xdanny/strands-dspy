"""DSPy module templates and signatures for common use cases."""

from strands_dspy.modules.programs import (
    ClassificationProgram,
    QAProgram,
    RAGProgram,
    ReasoningProgram,
)
from strands_dspy.modules.signatures import (
    Classification,
    ExtractionTask,
    QuestionAnswering,
    ReasoningTask,
    SummarizationTask,
)

__all__ = [
    # Programs
    "QAProgram",
    "ClassificationProgram",
    "ReasoningProgram",
    "RAGProgram",
    # Signatures
    "QuestionAnswering",
    "Classification",
    "ReasoningTask",
    "SummarizationTask",
    "ExtractionTask",
]
