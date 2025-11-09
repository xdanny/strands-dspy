"""Memory tools for querying DSPy optimization data from Strands agents."""

import json
import logging

from strands import tool
from strands.types.tools import ToolContext

from strands_dspy.storage.session_storage import DSPySessionStorage

logger = logging.getLogger(__name__)


@tool(
    name="retrieve_dspy_optimizations",
    description="Retrieve the latest optimized prompts from DSPy training",
    context=True,
)
async def retrieve_dspy_optimizations(
    tool_context: ToolContext,
    collection_agent_id: str = "dspy_training",
) -> str:
    """Get the most recent optimized prompts and instructions.

    This tool allows agents to query their own optimization history
    and see what improvements have been made to their prompts.

    Args:
        tool_context: Strands tool context (auto-injected)
        collection_agent_id: Agent ID where training data is stored
            (default: "dspy_training")

    Returns:
        JSON string containing optimized prompts and metadata
    """
    try:
        storage = DSPySessionStorage(tool_context.session_manager)

        # Get session ID from context
        session_id = getattr(tool_context.session_manager, "session_id", "default")

        # Retrieve latest optimization
        result = await storage.retrieve_latest_optimization(
            session_id=session_id,
            agent_id=collection_agent_id,
        )

        if not result:
            return json.dumps(
                {
                    "status": "no_optimizations",
                    "message": "No optimization results found. Train more examples to trigger optimization.",
                },
                indent=2,
            )

        # Format result for display
        output = {
            "status": "success",
            "optimizer": result.optimizer,
            "timestamp": result.timestamp,
            "train_size": result.train_size,
            "val_size": result.val_size,
            "best_score": result.best_score,
            "prompts": result.prompts,
        }

        if result.detailed_results:
            output["detailed_results"] = result.detailed_results

        return json.dumps(output, indent=2)

    except Exception as e:
        logger.error(f"Failed to retrieve optimizations: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to retrieve optimizations: {str(e)}",
            },
            indent=2,
        )


@tool(
    name="list_training_examples",
    description="List DSPy training examples that have been collected",
    context=True,
)
async def list_training_examples(
    limit: int = 10,
    min_score: float = 0.0,
    tool_context: ToolContext = None,
    collection_agent_id: str = "dspy_training",
) -> str:
    """List recent training examples with their scores.

    This tool shows what examples have been collected for prompt optimization.

    Args:
        limit: Maximum number of examples to return (default: 10)
        min_score: Minimum score filter (default: 0.0)
        tool_context: Strands tool context (auto-injected)
        collection_agent_id: Agent ID where training data is stored
            (default: "dspy_training")

    Returns:
        JSON string containing list of training examples
    """
    try:
        storage = DSPySessionStorage(tool_context.session_manager)

        # Get session ID from context
        session_id = getattr(tool_context.session_manager, "session_id", "default")

        # Retrieve examples
        examples = await storage.retrieve_training_examples(
            session_id=session_id,
            agent_id=collection_agent_id,
            min_score=min_score,
            limit=limit,
        )

        if not examples:
            return json.dumps(
                {
                    "status": "no_examples",
                    "message": "No training examples found. Examples are collected automatically from successful agent runs.",
                },
                indent=2,
            )

        # Format for display
        output = {
            "status": "success",
            "count": len(examples),
            "examples": [
                {
                    "inputs": ex.inputs,
                    "outputs": ex.outputs,
                    "score": ex.score,
                    "timestamp": ex.timestamp,
                    "metadata": ex.metadata,
                }
                for ex in examples
            ],
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        logger.error(f"Failed to list training examples: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to list training examples: {str(e)}",
            },
            indent=2,
        )


@tool(
    name="get_training_stats",
    description="Get statistics about collected DSPy training examples",
    context=True,
)
async def get_training_stats(
    tool_context: ToolContext,
    collection_agent_id: str = "dspy_training",
) -> str:
    """Get statistics about the training data collection.

    This tool provides an overview of how many examples have been collected,
    their score distribution, and other useful metrics.

    Args:
        tool_context: Strands tool context (auto-injected)
        collection_agent_id: Agent ID where training data is stored
            (default: "dspy_training")

    Returns:
        JSON string containing training statistics
    """
    try:
        storage = DSPySessionStorage(tool_context.session_manager)

        # Get session ID from context
        session_id = getattr(tool_context.session_manager, "session_id", "default")

        # Get stats
        stats = await storage.get_training_stats(
            session_id=session_id,
            agent_id=collection_agent_id,
        )

        # Get optimization history
        optimizations = await storage.retrieve_all_optimizations(
            session_id=session_id,
            agent_id=collection_agent_id,
        )

        output = {
            "status": "success",
            "training_examples": stats,
            "optimizations": {
                "count": len(optimizations),
                "history": [
                    {
                        "optimizer": opt.optimizer,
                        "timestamp": opt.timestamp,
                        "best_score": opt.best_score,
                        "train_size": opt.train_size,
                    }
                    for opt in optimizations
                ],
            },
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to get training stats: {str(e)}",
            },
            indent=2,
        )
