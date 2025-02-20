# RUN ME

This document guides you through running the solution accelerator.

We use poetry to manage dependencies and virtual environments. We also use it to manage the testing framework and run coverage reports automatically. See pyproject.toml for more details.

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
poetry run pytest -m "not slow"
```

5. Update dependencies:
```bash
poetry update
```

6. Install the IPython Kernel for Development
While inside the Poetry shell, install the IPython kernel for Jupyter:
```bash 
python -m ipykernel install --user --name=maud
```

# Format code with black
poetry run black maud/ tests/

# Sort imports with isort
poetry run isort maud/ tests/

# Run flake8 style checks
poetry run flake8 maud/ tests/

# Run mypy type checking
poetry run mypy maud/ tests/

## Development Testing of Gradio Interface

1. Run the gradio interface:
```bash
poetry run gradio app.py
```

2. Open the URL in your browser: http://localhost:7860/

## Logging

Using a dedicated logger for each module or class is generally more maintainable and flexible than using the root logger directly. This approach allows you to configure logging behavior (such as log levels and handlers) specifically for different parts of your application. 
