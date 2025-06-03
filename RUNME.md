# Running Multimodal Analysis of Unstructured Documents (MAUD)

This document guides your through running the solution accelerator, modifying it for your use case, and deploying it to Databricks.

## Quick Start

The solution accelerator is designed to be run on Databricks. Here are the steps to get started quickly.

1. Clone the repository

2. Spin up a Databricks cluster with ML Runtime 15.4 or higher

3. Start running the notebooks in order

## Modifications

We have tried to keep environment management simple. We use uv to manage the environment and the requirements. There is a single configuration file that is used to configure the entire solution in the root (`config.yaml`).

### Local Development
You can also test it locally. We use UV for environment management. Here are linux / macos instructions for setting up a local environment, but here is the [official guide](https://github.com/astral-sh/uv#installation).

```bash
brew install uv
```

```bash
uv venv .venv  
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Deployment

The intent of MAUD is to use cheap CPU compute in parallel to process many documents in a batch process.
