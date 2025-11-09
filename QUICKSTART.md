# Quick Start Guide

This guide will help you get started with `strands-dspy` using `uv` for package management and Gemini for testing.

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Gemini API key (get one at https://makersuite.google.com/app/apikey)

## Installation

### 1. Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/strands-agents/strands-dspy.git
cd strands-dspy

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with development dependencies
uv pip install -e ".[dev]"
```

### 3. Configure API Key

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_api_key_here
```

**⚠️ IMPORTANT**: Never commit `.env` to version control! It's already in `.gitignore`.

## Running Tests

### Run All Tests

```bash
uv run pytest
```

### Run with Coverage

```bash
uv run pytest --cov=strands_dspy --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Run Specific Tests

```bash
# Unit tests only
uv run pytest tests/test_storage.py

# Integration tests (uses real Gemini API)
uv run pytest tests/integration/

# Run with verbose output
uv run pytest -v
```

## Running Examples

### Basic Q&A Optimization

```bash
uv run python examples/basic_qa_optimization.py
```

This will:
1. Create a Strands agent with DSPy hooks
2. Run Q&A examples to collect training data
3. Automatically trigger optimization when threshold is reached
4. Store optimized prompts in session storage

### RAG Optimization

```bash
uv run python examples/rag_optimization.py
```

This demonstrates:
- Retrieval-augmented generation with Strands
- GEPA optimization for RAG workflows
- Custom feedback metrics

### Custom Metrics Guide

```bash
uv run python examples/custom_metric.py
```

Shows how to create custom evaluation metrics for:
- MIPRO (simple score-based)
- GEPA (feedback-based)
- Multi-criteria evaluation

## Basic Usage

```python
import os
import dspy
from dotenv import load_dotenv
from strands import Agent
from strands.session import FileSessionManager
from strands_dspy import (
    DSPyTrainingCollector,
    DSPyOptimizationHook,
    OptimizationConfig,
)

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure DSPy with Gemini
lm = dspy.LM(
    model="google/gemini-2.0-flash-lite",
    api_key=api_key,
    temperature=0.7
)
dspy.configure(lm=lm)

# Setup session manager
session_manager = FileSessionManager(session_id="my_session")

# Create training collector
collector = DSPyTrainingCollector(
    session_manager=session_manager,
    success_criteria=lambda r: (r.stop_reason == "end_turn", 1.0),
    input_extractor=lambda m: {"question": m[0]["content"][0]["text"]},
    output_extractor=lambda r: {"answer": r.message["content"][0]["text"]},
)

# Create optimizer
def create_program():
    class QA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.cot = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.cot(question=question)
    return QA()

def metric(example, prediction, trace=None):
    score = 1.0 if example.answer in prediction.answer else 0.0
    return dspy.Prediction(score=score, feedback="Match" if score else "No match")

optimizer = DSPyOptimizationHook(
    session_manager=session_manager,
    config=OptimizationConfig(
        optimizer_type="gepa",
        example_threshold=50
    ),
    metric=metric,
    program_factory=create_program,
)

# Create agent with hooks
agent = Agent(
    model="claude-3-5-sonnet-20241022",
    session_manager=session_manager,
    hooks=[collector, optimizer],
)

# Use the agent - optimization happens automatically!
result = agent("What is the capital of France?")
print(result.message)
```

## Development Workflow

### Format Code

```bash
uv run black src tests examples
```

### Lint

```bash
uv run ruff check src tests examples

# Auto-fix issues
uv run ruff check --fix src tests examples
```

### Type Check

```bash
uv run mypy src
```

### Run All Quality Checks

```bash
uv run black src tests examples && \
uv run ruff check src tests examples && \
uv run mypy src && \
uv run pytest --cov=strands_dspy
```

## Project Structure

```
strands-dspy/
├── src/strands_dspy/       # Main package
├── tests/                  # Test suite
│   ├── integration/        # Integration tests (uses real API)
│   └── test_*.py          # Unit tests
├── examples/               # Usage examples
├── .env.example            # Example environment variables
├── .env                    # Your API keys (DO NOT COMMIT)
└── pyproject.toml          # Package configuration
```

## Troubleshooting

### API Key Issues

If you see `GEMINI_API_KEY not found`:

1. Make sure `.env` exists and contains your API key
2. Verify `.env` is in the project root directory
3. Check that the key is valid at https://makersuite.google.com/

### Import Errors

If you see import errors:

```bash
# Reinstall in editable mode
uv pip install -e ".[dev]"
```

### Test Failures

If tests fail with API errors:

1. Check your API key is valid
2. Verify you have API quota remaining
3. Try running tests individually: `pytest tests/test_storage.py -v`

## Next Steps

- Read the [full README](README.md) for detailed documentation
- Check out the [contributing guide](CONTRIBUTING.md)
- Explore the [examples](examples/) directory
- Join the Strands community discussions

## Support

- GitHub Issues: https://github.com/strands-agents/strands-dspy/issues
- Strands Documentation: https://strandsagents.com
- DSPy Documentation: https://dspy-docs.vercel.app
