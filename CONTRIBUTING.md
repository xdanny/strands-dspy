# Contributing to strands-dspy

Thank you for your interest in contributing to strands-dspy! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/strands-dspy.git
cd strands-dspy
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with all development dependencies.

## Code Quality Standards

### Formatting

We use `black` for code formatting:

```bash
uv run black src tests examples
```

### Linting

We use `ruff` for linting:

```bash
uv run ruff check src tests examples
```

To auto-fix issues:

```bash
uv run ruff check --fix src tests examples
```

### Type Checking

We use `mypy` for type checking:

```bash
uv run mypy src
```

## Testing

### Running Tests

Run all tests:

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=strands_dspy --cov-report=html
```

View coverage report:

```bash
open htmlcov/index.html  # On macOS
# or
xdg-open htmlcov/index.html  # On Linux
# or
start htmlcov/index.html  # On Windows
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names: `test_<what>_<when>_<expected>`
- Use fixtures from `conftest.py` for common test data
- Aim for >80% code coverage

Example:

```python
def test_store_training_example_creates_message(mock_session_manager):
    """Test that storing a training example creates a session message."""
    storage = DSPySessionStorage(mock_session_manager)

    await storage.store_training_example(
        session_id="test",
        agent_id="test_agent",
        inputs={"question": "test"},
        outputs={"answer": "test"},
        score=1.0
    )

    assert mock_session_manager.create_message.called
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clear, concise commit messages
- Add tests for new functionality
- Update documentation as needed
- Follow existing code style

### 3. Run Quality Checks

Before submitting:

```bash
# Format code
uv run black src tests examples

# Lint
uv run ruff check src tests examples

# Type check
uv run mypy src

# Run tests
uv run pytest --cov=strands_dspy
```

### 4. Submit Pull Request

- Push your branch to GitHub
- Create a pull request against `main`
- Fill out the PR template
- Link any related issues
- Wait for CI checks to pass
- Address review feedback

## Code Review Guidelines

When reviewing PRs:

- Check code quality and style
- Verify tests are comprehensive
- Ensure documentation is updated
- Test the changes locally
- Provide constructive feedback

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of the function.

    More detailed explanation if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        ```python
        result = function_name("test", 42)
        ```
    """
    pass
```

### README Updates

When adding features:

- Update the main README.md
- Add usage examples
- Update the API reference section
- Add to the feature list

### Examples

When adding new functionality:

- Create example scripts in `examples/`
- Include comprehensive comments
- Show both basic and advanced usage
- Test examples before submitting

## Release Process

(For maintainers)

### 1. Update Version

Update version in:
- `src/strands_dspy/__init__.py`
- `pyproject.toml`

### 2. Update Changelog

Add release notes to `CHANGELOG.md` (create if needed):

```markdown
## [0.2.0] - 2024-01-15

### Added
- New feature X
- Support for Y

### Changed
- Improved performance of Z

### Fixed
- Bug in component A
```

### 3. Create Release

```bash
git tag v0.2.0
git push origin v0.2.0
```

Create a GitHub release, which will trigger the publish workflow.

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

### Getting Help

- Check existing issues and discussions
- Ask questions in GitHub Discussions
- Join the Strands community channels
- Read the documentation first

### Reporting Issues

When reporting bugs:

1. Check if the issue already exists
2. Provide a minimal reproduction case
3. Include relevant logs and error messages
4. Specify your environment (OS, Python version, etc.)
5. Use issue templates when available

### Suggesting Features

When suggesting features:

1. Check if it's already been suggested
2. Explain the use case and benefits
3. Provide examples of how it would work
4. Consider implementation complexity
5. Be open to discussion and alternatives

## Project Structure

```
strands-dspy/
├── src/strands_dspy/       # Main package code
│   ├── hooks/              # Hook implementations
│   ├── optimizers/         # MIPRO/GEPA wrappers
│   ├── storage/            # SessionManager utilities
│   ├── tools/              # Memory tools
│   ├── modules/            # DSPy templates
│   └── types.py            # Type definitions
├── tests/                  # Test suite
├── examples/               # Usage examples
├── .github/workflows/      # CI/CD workflows
└── docs/                   # Documentation (future)
```

## Additional Resources

- [Strands Documentation](https://strandsagents.com)
- [DSPy Documentation](https://dspy-docs.vercel.app)
- [Community Packages Guide](https://strandsagents.com/latest/documentation/docs/community/community-packages/)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open a discussion or issue if you have any questions!
