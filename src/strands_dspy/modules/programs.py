"""Common DSPy program templates for typical agent tasks."""

import dspy

from strands_dspy.modules.signatures import (
    Classification,
    QuestionAnswering,
    ReasoningTask,
)


class QAProgram(dspy.Module):
    """Question-answering program using chain-of-thought reasoning.

    This is a simple but effective template for QA tasks that can be optimized
    with DSPy's MIPRO or GEPA.

    Example:
        ```python
        program = QAProgram()
        result = program(question="What is the capital of France?")
        print(result.answer)  # "Paris"
        ```
    """

    def __init__(self):
        super().__init__()
        self.answer_generator = dspy.ChainOfThought(QuestionAnswering)

    def forward(self, question: str, context: str = ""):
        """Generate an answer to the question.

        Args:
            question: The question to answer
            context: Optional context or background information

        Returns:
            dspy.Prediction with answer field
        """
        result = self.answer_generator(question=question, context=context)
        return dspy.Prediction(answer=result.answer, reasoning=result.rationale)


class ClassificationProgram(dspy.Module):
    """Text classification program with confidence scoring.

    Classifies input text into predefined categories with confidence estimation.

    Example:
        ```python
        program = ClassificationProgram()
        result = program(
            text="I love this product!",
            categories="positive,negative,neutral"
        )
        print(result.category)    # "positive"
        print(result.confidence)  # "high"
        ```
    """

    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(Classification)

    def forward(self, text: str, categories: str):
        """Classify the text into one of the categories.

        Args:
            text: The text to classify
            categories: Comma-separated list of categories

        Returns:
            dspy.Prediction with category and confidence fields
        """
        result = self.classifier(text=text, categories=categories)
        return dspy.Prediction(
            category=result.category,
            confidence=result.confidence,
            reasoning=result.rationale,
        )


class ReasoningProgram(dspy.Module):
    """Multi-step reasoning program for complex problems.

    Uses chain-of-thought to break down problems and reason through solutions.

    Example:
        ```python
        program = ReasoningProgram()
        result = program(
            problem="How many tennis balls can fit in a school bus?",
            constraints="Use metric measurements"
        )
        print(result.reasoning)  # Step-by-step breakdown
        print(result.solution)   # Final answer
        ```
    """

    def __init__(self):
        super().__init__()
        self.reasoner = dspy.ChainOfThought(ReasoningTask)

    def forward(self, problem: str, constraints: str = ""):
        """Solve a problem using step-by-step reasoning.

        Args:
            problem: The problem to solve
            constraints: Any constraints or requirements

        Returns:
            dspy.Prediction with reasoning and solution fields
        """
        result = self.reasoner(problem=problem, constraints=constraints)
        return dspy.Prediction(reasoning=result.reasoning, solution=result.solution)


class RAGProgram(dspy.Module):
    """Retrieval-Augmented Generation program.

    Combines retrieval with generation for knowledge-intensive tasks.

    Example:
        ```python
        # Requires a retriever to be configured
        dspy.settings.configure(rm=your_retriever)

        program = RAGProgram(k=3)
        result = program(question="What is quantum entanglement?")
        print(result.answer)
        ```
    """

    def __init__(self, k: int = 3):
        """Initialize RAG program.

        Args:
            k: Number of documents to retrieve
        """
        super().__init__()
        self.k = k
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought(QuestionAnswering)

    def forward(self, question: str):
        """Answer a question using retrieved context.

        Args:
            question: The question to answer

        Returns:
            dspy.Prediction with answer and context fields
        """
        # Retrieve relevant passages
        retrieval_result = self.retrieve(question)
        context = "\n".join(retrieval_result.passages)

        # Generate answer with context
        result = self.generate(question=question, context=context)

        return dspy.Prediction(
            answer=result.answer, context=context, reasoning=result.rationale
        )
