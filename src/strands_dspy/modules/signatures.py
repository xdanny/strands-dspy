"""Common DSPy signatures for typical agent tasks."""

import dspy


class QuestionAnswering(dspy.Signature):
    """Answer questions accurately and concisely based on the given context or knowledge."""

    question: str = dspy.InputField(desc="The question to answer")
    context: str = dspy.InputField(desc="Relevant context or background information", default="")
    answer: str = dspy.OutputField(desc="The answer to the question")


class Classification(dspy.Signature):
    """Classify the input text into one of the predefined categories."""

    text: str = dspy.InputField(desc="The text to classify")
    categories: str = dspy.InputField(desc="Comma-separated list of possible categories")
    category: str = dspy.OutputField(desc="The selected category")
    confidence: str = dspy.OutputField(desc="Confidence level (high/medium/low)")


class ReasoningTask(dspy.Signature):
    """Perform step-by-step reasoning to solve a complex problem."""

    problem: str = dspy.InputField(desc="The problem to solve")
    constraints: str = dspy.InputField(desc="Any constraints or requirements", default="")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process")
    solution: str = dspy.OutputField(desc="The final solution")


class SummarizationTask(dspy.Signature):
    """Summarize the given text concisely while preserving key information."""

    text: str = dspy.InputField(desc="The text to summarize")
    max_length: str = dspy.InputField(desc="Maximum length guidance", default="3-5 sentences")
    summary: str = dspy.OutputField(desc="Concise summary of the text")


class ExtractionTask(dspy.Signature):
    """Extract specific information from the given text."""

    text: str = dspy.InputField(desc="The text to extract information from")
    extraction_target: str = dspy.InputField(desc="What to extract (e.g., 'dates', 'names', 'key facts')")
    extracted_info: str = dspy.OutputField(desc="The extracted information")
