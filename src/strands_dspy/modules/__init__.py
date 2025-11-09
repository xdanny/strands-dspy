"""DSPy module templates and signatures for common use cases."""

from strands_dspy.modules.programs import (
    QAProgram,
    ClassificationProgram,
    ReasoningProgram,
    RAGProgram,
)
from strands_dspy.modules.signatures import (
    QuestionAnswering,
    Classification,
    ReasoningTask,
    SummarizationTask,
    ExtractionTask,
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
