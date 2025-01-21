# RUN ME

This document guides you through running the solution accelerator.

## Development Setup with Poetry

### Quick Start

1. Install Poetry following the [official guide](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
poetry env use 3.11
poetry install
poetry shell  # Activate the virtual environment
```

### Key Poetry Commands

- Add dependency: `poetry add package-name`
- Add dev dependency: `poetry add --group dev package-name`
- Run tests: `poetry run pytest`
- Update dependencies: `poetry update`

### DAB Development

For Data Analysis Bundle (DAB) development:

1. Ensure dependencies are in `pyproject.toml`
2. Set Python version:
```bash
poetry env use 3.11
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Run tests:
```bash
poetry run pytest
```

5. Update dependencies:
```bash
poetry update
```

# Format code with black
poetry run black src/ tests/

# Sort imports with isort
poetry run isort src/ tests/

# Run flake8 style checks
poetry run flake8 src/ tests/

# Run mypy type checking
poetry run mypy src/ tests/