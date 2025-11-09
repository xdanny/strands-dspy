"""Test configuration for strands-dspy.

Sets up Gemini API and DSPy for testing.
"""

import os

import dspy
from dotenv import load_dotenv


def setup_test_env():
    """Setup test environment with OpenRouter API.

    Loads API key from .env and configures DSPy with MiniMax M2 (free on OpenRouter).
    """
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment. "
            "Please copy .env.example to .env and add your API key."
        )

    # Configure DSPy with OpenRouter MiniMax M2 (free tier)
    lm = dspy.LM(
        model="openrouter/minimax/minimax-m2:free",
        api_key=api_key,
        temperature=0.0,  # Deterministic for testing
    )

    dspy.configure(lm=lm)

    return lm


def get_test_lm():
    """Get configured test language model.

    Returns:
        dspy.LM configured for OpenRouter MiniMax M2
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    return dspy.LM(
        model="openrouter/minimax/minimax-m2:free",
        api_key=api_key,
        temperature=0.0,
    )
