"""
RAG (Retrieval-Augmented Generation) optimization example.

This example demonstrates:
1. Optimizing RAG agents with DSPy
2. Using retrieval in Strands agents
3. Evaluating retrieval quality
4. GEPA optimization for RAG workflows
"""

import asyncio
import os

import dspy
from dotenv import load_dotenv
from strands import Agent, tool
from strands.session import FileSessionManager

from strands_dspy import (
    DSPyOptimizationHook,
    DSPyTrainingCollector,
    OptimizationConfig,
)

# Load environment variables
load_dotenv()


# Simulated knowledge base for demo
KNOWLEDGE_BASE = {
    "python": [
        "Python is a high-level programming language known for its readability.",
        "Python was created by Guido van Rossum and released in 1991.",
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional.",
    ],
    "dspy": [
        "DSPy is a framework for programming language models.",
        "DSPy provides optimizers like MIPRO and GEPA for prompt engineering.",
        "DSPy uses signatures to define input-output specifications.",
    ],
    "strands": [
        "Strands is an agent framework built on Claude.",
        "Strands provides hooks for extending agent behavior.",
        "Strands uses SessionManager for conversation persistence.",
    ],
}


@tool(name="search_knowledge", description="Search the knowledge base for relevant information")
def search_knowledge(query: str, top_k: int = 3) -> str:
    """Simple keyword-based search tool.

    In production, you'd use vector search, BM25, or hybrid retrieval.
    """
    query_lower = query.lower()
    results = []

    # Simple keyword matching
    for topic, passages in KNOWLEDGE_BASE.items():
        for passage in passages:
            if any(word in passage.lower() for word in query_lower.split()):
                results.append({"passage": passage, "topic": topic})

    # Return top-k
    results = results[:top_k]

    if not results:
        return "No relevant information found."

    # Format results
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"[{i}] {result['passage']}")

    return "\n".join(formatted)


# DSPy RAG Program
class RAGProgram(dspy.Module):
    """Simple RAG program for demonstration."""

    def __init__(self):
        super().__init__()

        class RAGSignature(dspy.Signature):
            """Answer questions using retrieved context."""

            question: str = dspy.InputField()
            context: str = dspy.InputField()
            answer: str = dspy.OutputField()

        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question: str, context: str = ""):
        """Generate answer with context."""
        result = self.generate(question=question, context=context)
        return dspy.Prediction(answer=result.answer, reasoning=result.rationale)


def create_rag_program():
    """Factory for creating RAG programs."""
    return RAGProgram()


def rag_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> dspy.Prediction:
    """Metric for evaluating RAG quality.

    Evaluates both answer correctness and context usage.
    """
    feedback_parts = []
    scores = {}

    # Check answer correctness
    if example.answer.lower() in prediction.answer.lower():
        scores["correctness"] = 1.0
        feedback_parts.append("✓ Answer is correct")
    else:
        scores["correctness"] = 0.0
        feedback_parts.append(f"✗ Answer is incorrect. Expected: {example.answer}")

    # Check if reasoning uses the context (if available)
    if hasattr(prediction, "reasoning") and hasattr(example, "context"):
        context_terms = set(example.context.lower().split())
        reasoning_terms = set(prediction.reasoning.lower().split())
        overlap = context_terms.intersection(reasoning_terms)

        if len(overlap) >= 3:
            scores["context_usage"] = 1.0
            feedback_parts.append("✓ Reasoning uses retrieved context effectively")
        else:
            scores["context_usage"] = 0.3
            feedback_parts.append("✗ Reasoning doesn't sufficiently use the retrieved context")
    else:
        scores["context_usage"] = 1.0  # Neutral if context not available

    # Overall score
    total_score = (scores["correctness"] * 0.7) + (scores["context_usage"] * 0.3)

    feedback = "\n".join(feedback_parts)
    return dspy.Prediction(score=total_score, feedback=feedback)


def success_criteria(result):
    """Evaluate if a RAG agent run was successful."""
    if result.stop_reason == "end_turn":
        return True, 1.0
    return False, 0.0


def extract_inputs(messages):
    """Extract question from messages."""
    if messages and len(messages) > 0:
        first_msg = messages[0]
        if "content" in first_msg and len(first_msg["content"]) > 0:
            content_block = first_msg["content"][0]
            if isinstance(content_block, dict) and "text" in content_block:
                # In RAG, we also need to capture the context from tool results
                # For simplicity, we'll extract just the question here
                return {"question": content_block["text"]}

    return {"question": ""}


def extract_outputs(result):
    """Extract answer from agent result."""
    if result.message and "content" in result.message:
        content = result.message["content"]
        if len(content) > 0 and isinstance(content[0], dict):
            return {"answer": content[0].get("text", "")}

    return {"answer": ""}


async def main():
    """Run RAG optimization example."""
    print("=" * 60)
    print("strands-dspy: RAG Optimization Example")
    print("=" * 60)

    # Setup session manager
    session_manager = FileSessionManager(session_id="rag_example")
    print("\n✓ Created session manager")

    # Create collector
    collector = DSPyTrainingCollector(
        session_manager=session_manager,
        success_criteria=success_criteria,
        input_extractor=extract_inputs,
        output_extractor=extract_outputs,
    )
    print("✓ Created training collector")

    # Create optimizer with GEPA (better for RAG feedback)
    optimizer = DSPyOptimizationHook(
        session_manager=session_manager,
        config=OptimizationConfig(
            optimizer_type="gepa",
            num_candidates=5,
            num_trials=10,
            auto_budget="light",
        ),
        metric=rag_metric,
        program_factory=create_rag_program,
        example_threshold=8,  # Low for demo
    )
    print("✓ Created GEPA optimizer for RAG")

    # Create agent with retrieval tool
    agent = Agent(
        model="claude-3-5-sonnet-20241022",
        system_prompt=(
            "You are a helpful assistant with access to a knowledge base. "
            "When answering questions, first search for relevant information, "
            "then provide accurate answers based on the retrieved context."
        ),
        tools=[search_knowledge],
        session_manager=session_manager,
        hooks=[collector, optimizer],
    )
    print("✓ Created RAG agent with search tool\n")

    # Example questions
    questions = [
        "What is Python?",
        "Who created Python?",
        "What is DSPy?",
        "What are DSPy optimizers?",
        "What is Strands?",
        "How does Strands handle conversations?",
        "What programming paradigms does Python support?",
        "What is a DSPy signature?",
    ]

    print("Running RAG queries...")
    print("-" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Question: {question}")

        # Invoke agent (it will use the search tool automatically)
        result = await agent.invoke_async(question)

        # Extract answer
        answer = ""
        if result.message and "content" in result.message:
            content = result.message["content"]
            if len(content) > 0 and isinstance(content[0], dict):
                answer = content[0].get("text", "")

        print(f"         Answer: {answer[:100]}...")

    print("\n" + "=" * 60)
    print("RAG optimization example complete!")
    print("=" * 60)
    print("\nOptimization will improve:")
    print("  - How the agent uses retrieved context")
    print("  - When to trigger retrieval")
    print("  - How to synthesize answers from multiple sources")
    print("\nSession data: sessions/rag_example/")


if __name__ == "__main__":
    # Configure DSPy with Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please copy .env.example to .env and add your Gemini API key.")
        exit(1)

    # Configure DSPy with Gemini 2.5 Flash Lite
    lm = dspy.LM(
        model="google/gemini-2.0-flash-lite",
        api_key=api_key,
        temperature=0.7,
    )
    dspy.configure(lm=lm)

    print("✓ Configured DSPy with Gemini 2.5 Flash Lite\n")

    asyncio.run(main())
