# strands-dspy

**DSPy prompt optimization integration for Strands agents**

Automatically improve your Strands agent prompts using [DSPy](https://github.com/stanfordnlp/dspy)'s powerful optimization algorithms (MIPRO and GEPA).

## Features

- 🎯 **Automatic Prompt Optimization**: Leverage DSPy's MIPRO and GEPA optimizers to improve agent prompts based on successful runs
- 🎁 **Pre-built Helpers**: Common extractors, metrics, and success criteria - no boilerplate needed
- 📋 **Copy-Paste Blueprints**: Working templates for Q&A and classification ready to customize
- 📊 **Training Example Collection**: Automatically collect and store training examples from agent executions
- 🔄 **Seamless Integration**: Hook-based architecture integrates with Strands agents
- 💾 **Native Memory Storage**: Uses Strands' SessionManager for all data persistence (no custom databases needed)

## Installation

### From PyPI (when published)

```bash
pip install strands-dspy
# or with uv (recommended)
uv pip install strands-dspy
```

### From Source

```bash
git clone https://github.com/strands-agents/strands-dspy.git
cd strands-dspy

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Setup API Key

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_api_key_here
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.**

## Quick Start

### Simple Q&A Optimization (Using Pre-built Helpers)

```python
import os
import dspy
from dotenv import load_dotenv
from strands import Agent
from strands_dspy import (
    TrainingCollector,
    GEPAOptimizer,
    SessionStorageBackend,
)
from strands_dspy.helpers import (
    extract_first_user_text,
    extract_last_assistant_text,
    end_turn_success,
    contains_match_gepa,
)
from strands_dspy.types import OptimizationConfig

load_dotenv()

# 1. Configure DSPy LM
lm = dspy.LM(
    model="openrouter/minimax/minimax-m2:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
dspy.configure(lm=lm)

# 2. Create agent
agent = Agent(
    name="qa-agent",
    instructions="You are a helpful Q&A assistant.",
    model="gemini/gemini-2.5-flash-lite",
)

# 3. Set up training collection (using pre-built helpers!)
storage = SessionStorageBackend()
collector = TrainingCollector(
    storage=storage,
    input_extractor=extract_first_user_text,     # Pre-built!
    output_extractor=extract_last_assistant_text, # Pre-built!
    success_criteria=end_turn_success,            # Pre-built!
)
collector.attach(agent)

# 4. Collect training examples
training_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    # ... more examples
]

for pair in training_data:
    agent.invoke(pair["question"])

# 5. Run optimization
examples = storage.get_training_examples("session-1", "qa-agent")
trainset = [
    dspy.Example(question=ex.inputs["question"], answer=ex.outputs["answer"]).with_inputs("question")
    for ex in examples
]

config = OptimizationConfig(
    optimizer_type="gepa",
    auto_budget="light",
)

optimizer = GEPAOptimizer(
    config=config,
    metric=contains_match_gepa,  # Pre-built GEPA metric!
)

# Define your DSPy program
class QAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate(question=question)

optimized, result = optimizer.optimize(
    program=QAProgram(),
    trainset=trainset,
)

print(f"Best score: {result.best_score:.2%}")
```

**💡 See [`examples/blueprints/`](examples/blueprints/) for complete copy-paste templates!**

## Pre-built Helpers

To make setup easier, `strands-dspy` includes pre-built helper functions for common patterns:

### Extractors
Extract structured data from Strands messages:

```python
from strands_dspy.helpers import (
    extract_first_user_text,        # First user message as "question"
    extract_last_assistant_text,    # Last assistant message as "answer"
    extract_all_user_messages,      # All user messages as list
    extract_field_from_message,     # Custom field extraction
    combine_extractors,             # Merge multiple extractors
)

# Example: Basic Q&A
collector = TrainingCollector(
    storage=storage,
    input_extractor=extract_first_user_text,
    output_extractor=extract_last_assistant_text,
    success_criteria=end_turn_success,
)
```

### Metrics
Pre-built metrics for common tasks:

```python
from strands_dspy.helpers import (
    exact_match,           # Exact string match
    contains_match,        # Substring match (more lenient)
    numeric_match,         # Number extraction and comparison
    make_gepa_metric,      # Convert simple metric to GEPA format
    # Pre-made GEPA versions:
    exact_match_gepa,
    contains_match_gepa,
    numeric_match_gepa,
)

# Example: MIPRO with simple metric
optimizer = MIPROOptimizer(
    config=config,
    metric=exact_match,  # Returns float
)

# Example: GEPA with feedback
optimizer = GEPAOptimizer(
    config=config,
    metric=contains_match_gepa,  # Returns Prediction(score, feedback)
)

# Example: Custom metric with GEPA
def my_metric(example, prediction, trace=None):
    return 1.0 if example.answer == prediction.answer else 0.0

gepa_metric = make_gepa_metric(my_metric)
```

### Success Criteria
Pre-built functions for filtering training examples:

```python
from strands_dspy.helpers import (
    always_success,             # Collect all examples
    end_turn_success,           # Only successful completions
    no_error_success,           # Only error-free runs
    min_length_success,         # Minimum response length
    combine_criteria,           # AND multiple criteria
)

# Example: Collect only complete, error-free responses
collector = TrainingCollector(
    storage=storage,
    success_criteria=combine_criteria(
        end_turn_success,
        no_error_success,
        min_length_success(min_chars=50),
    ),
    ...
)
```

## Blueprints

Copy-paste ready templates in [`examples/blueprints/`](examples/blueprints/):

- **[`qa_blueprint.py`](examples/blueprints/qa_blueprint.py)** - Complete Q&A agent with GEPA optimization
- **[`classification_blueprint.py`](examples/blueprints/classification_blueprint.py)** - Text classification with MIPRO

Each blueprint is a fully working example you can:
1. Copy to your project
2. Update the training data
3. Run immediately

## How It Works

### 1. Training Collection

The `DSPyTrainingCollector` hook:
- Monitors agent invocations via Strands lifecycle hooks
- Evaluates results using your success criteria function
- Stores successful runs as training examples in SessionManager
- Supports custom input/output extractors for flexible data shapes

### 2. Automatic Optimization

The `DSPyOptimizationHook` hook:
- Monitors training example count
- Triggers DSPy optimization when threshold is reached
- Supports both MIPRO (instruction optimization) and GEPA (grounded feedback optimization)
- Stores optimized prompts back to SessionManager

### 3. Storage Architecture

All data is stored using Strands' native `SessionManager`:

```
sessions/
  └── user_123/
      └── agents/
          └── agent_dspy_training/
              ├── agent.json
              └── messages/
                  ├── 0.json    # Training example 1
                  ├── 1.json    # Training example 2
                  ├── ...
                  └── 50.json   # Optimized prompts
```

Training examples are stored as `assistant` role messages with JSON content.
Optimized prompts are stored as `system` role messages.

## Optimizers

### MIPRO (Multi-prompt Instruction Proposal Optimizer)

Best for: General prompt optimization with automatic instruction generation

```python
optimizer = DSPyOptimizationHook(
    session_manager=session_manager,
    config=OptimizationConfig(
        optimizer_type="mipro",
        num_candidates=10,
        num_trials=30,
        auto_budget="light",  # or "medium", "heavy"
    ),
    metric=your_metric,
    program_factory=create_program
)
```

### GEPA (Grounded Proposal)

Best for: Rich feedback-based optimization with domain-specific guidance

```python
optimizer = DSPyOptimizationHook(
    session_manager=session_manager,
    config=OptimizationConfig(
        optimizer_type="gepa",
        num_candidates=10,
        num_trials=30,
        auto_budget="light",
        track_stats=True,  # Track detailed optimization stats
    ),
    metric=your_feedback_metric,  # Must return dspy.Prediction(score, feedback)
    program_factory=create_program
)
```

## Configuration Options

### OptimizationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_type` | `"mipro"` \| `"gepa"` | Required | Which optimizer to use |
| `num_candidates` | `int` | `10` | Number of instruction candidates to generate |
| `num_trials` | `int` | `30` | Number of optimization trials to run |
| `auto_budget` | `"light"` \| `"medium"` \| `"heavy"` | `"light"` | Preset optimization budget |
| `minibatch` | `bool` | `True` | Use minibatch evaluation |
| `minibatch_size` | `int` | `25` | Size of evaluation minibatches |
| `track_stats` | `bool` | `True` | Track detailed optimization statistics |

### DSPyTrainingCollector

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_manager` | `SessionManager` | Strands session manager instance |
| `success_criteria` | `Callable` | Function returning `(is_success: bool, score: float)` |
| `input_extractor` | `Callable` | Function extracting inputs from messages |
| `output_extractor` | `Callable` | Function extracting outputs from agent result |
| `collection_agent_id` | `str` | Agent ID for storing training data (default: `"dspy_training"`) |

### DSPyOptimizationHook

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_manager` | `SessionManager` | Strands session manager instance |
| `config` | `OptimizationConfig` | Optimization configuration |
| `metric` | `Callable` | DSPy metric function |
| `program_factory` | `Callable` | Factory function creating DSPy program |
| `example_threshold` | `int` | Number of examples before triggering optimization |
| `collection_agent_id` | `str` | Agent ID where training data is stored |

## Memory Tools

Query optimization results and training data:

```python
from strands_dspy.tools import retrieve_dspy_optimizations, list_training_examples

# Add tools to your agent
agent = Agent(
    model="claude-3-5-sonnet-20241022",
    tools=[retrieve_dspy_optimizations, list_training_examples],
    session_manager=session_manager
)

# Agent can now query its own optimization history
result = agent("Show me the latest optimized prompts")
```

## Examples

### Blueprints (Recommended Starting Point)

See [`examples/blueprints/`](examples/blueprints/) for copy-paste ready templates:

- **[`qa_blueprint.py`](examples/blueprints/qa_blueprint.py)** - Complete Q&A agent with GEPA optimization and pre-built helpers
- **[`classification_blueprint.py`](examples/blueprints/classification_blueprint.py)** - Text classification with MIPRO and custom metrics

### Additional Examples

See [`examples/`](examples/) for more advanced examples:

- Integration tests demonstrating real optimization improvements
- Custom metric implementations
- Advanced extractor patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Strands Agent                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Lifecycle Hooks                          │  │
│  │  ┌──────────────────┐  ┌──────────────────────────┐  │  │
│  │  │ Training         │  │ Optimization             │  │  │
│  │  │ Collector        │  │ Hook                     │  │  │
│  │  │                  │  │                          │  │  │
│  │  │ • Monitor runs   │  │ • Check threshold        │  │  │
│  │  │ • Extract data   │  │ • Run MIPRO/GEPA         │  │  │
│  │  │ • Store examples │  │ • Apply optimizations    │  │  │
│  │  └────────┬─────────┘  └────────────┬─────────────┘  │  │
│  └───────────┼────────────────────────┼────────────────┘  │
└──────────────┼────────────────────────┼───────────────────┘
               │                        │
               ▼                        ▼
    ┌──────────────────────────────────────────────┐
    │         SessionManager Storage               │
    │  • Training examples (assistant messages)    │
    │  • Optimized prompts (system messages)       │
    │  • Metadata and statistics                   │
    └──────────────────────────────────────────────┘
```

## Development

### Setup

```bash
git clone https://github.com/strands-agents/strands-dspy.git
cd strands-dspy
pip install -e ".[dev]"
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format code
uv run black src tests examples

# Lint
uv run ruff check src tests examples

# Type check
uv run mypy src
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details

## Links

- [Strands Documentation](https://strandsagents.com)
- [DSPy Documentation](https://dspy-docs.vercel.app)
- [Community Packages Guide](https://strandsagents.com/latest/documentation/docs/community/community-packages/)

## Citation

If you use strands-dspy in your research, please cite:

```bibtex
@software{strands_dspy,
  title = {strands-dspy: DSPy Integration for Strands Agents},
  author = {Strands Community},
  year = {2024},
  url = {https://github.com/strands-agents/strands-dspy}
}
```
