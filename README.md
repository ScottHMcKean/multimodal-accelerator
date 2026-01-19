# Multimodal Document Processing - Simplified

[![DBR](https://img.shields.io/badge/DBR-15.4_LTS_ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/CHANGE_ME.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=databricks&style=for-the-badge)](https://databricks.com/try-databricks)

**Three streamlined approaches for document processing at scale.**

## 🎯 Quick Start

Choose your processing method:

### 1. **AI_PARSE** - Databricks Native (Simplest)
```sql
-- Zero setup, works with serverless
SELECT path, ai_parse(content) as parsed 
FROM read_files('/Volumes/main/default/raw_docs/*.pdf')
```

### 2. **Docling + Ray** - Parallel Processing (Flexible)  
```bash
uv run python convert_docling_ray.py
```

### 3. **Docling Serving** - GPU + VLM (Most Advanced)
```bash
uv run python deploy_docling_endpoint.py  # Deploy
uv run python convert_docling_serving.py  # Process with VLM
```

## 📊 Method Comparison

| Method | Complexity | Setup | VLM Support | Best For |
|--------|------------|-------|-------------|----------|
| **AI_PARSE** | Simple | None | No | Large batches, serverless |
| **Docling + Ray** | Medium | Ray cluster | Optional | Custom processing |
| **Docling Serving** | Low | Model endpoint | Native | GPU + VLM features |

## 🚀 Benefits

✅ **90% less complexity** - Three focused methods instead of complex pipeline  
✅ **Native VLM support** - Granite Vision, SmolVLM built-in  
✅ **Choose your approach** - Pick the right tool for each use case  
✅ **Standard implementations** - No custom code to maintain  

See [README_SIMPLIFIED.md](README_SIMPLIFIED.md) for detailed documentation.

4. Infer: Use a foundation model and agent framework to search and extract information from the documents.

5. Interface: Provide a basic user interface for interacting with the agent and gathering feedback.

<img src="assets/Multimodal Reference Architecture.png" width="800px">

This architecture leverages several services from the Databricks platform: Databricks Apps to deliver a user interface, Databricks Vector Search to serve the retriever, Mosaic AI Gateway to serve the foundation model, Delta Live Tables to sync and maintain the vector database, and Model Serving to serve the agent framework, as show in the the process diagram below.

<img src="assets/Multimodal Process Flow.png" width="800px">

## Key Services and Costs

| Service | Example Cost* | Latency | Reference |
|---------|------------|---------------|----------|
| Databricks Apps |  $180/month | <100ms | [Apps Pricing](https://www.databricks.com/product/pricing) |
| Mosaic Vector Search  | $250/month | 10-100ms | [Docs](https://docs.databricks.com/en/generative-ai/vector-search.html) |
| Mosaic AI Gateway  | $1.00/1M tokens | 500-5000ms | [Docs](https://docs.databricks.com/en/machine-learning/ai-gateway/index.html) |
| Mosaic AI Model Serving  | $250/month | ~100ms | [Docs](https://docs.databricks.com/en/machine-learning/model-serving/index.html) |

\* Example costs are illustrative estimates only and will vary based on usage, region, and implementation details. DBU = Databricks Unit.

# Running Multimodal Analysis of Unstructured Documents (MAUD)

This document guides your through running the solution accelerator, modifying it for your use case, and deploying it to Databricks.

## Quick Start

The solution accelerator is designed to be run on Databricks. Here are the steps to get started quickly.

1. Clone the repository

2. Spin up a Databricks cluster with ML Runtime 15.4 or higher

3. Start running the notebooks in order

MAUD uses the LangGraph library to build and deploy agent workflows. This documentation is a summary of the LangGraph documentation and the code in the repository.

## LangGraph
The [graph definitions](https://langchain-ai.github.io/langgraph/reference/graphs/#graph-definitions) in the LangGraph documentation are helpful for understanding the options for constructing workflows.

LangGraph offers multiple streaming modes, but this repo uses the basic [stream](https://langchain-ai.github.io/langgraph/concepts/streaming/#streaming) mode since MLflow doesn't support async yet. This allows returning node outputs to users during execution, enabling feedback during intermediate steps.

## Modifications

We have tried to keep environment management simple. We use uv to manage the environment and the requirements. There is a single configuration file that is used to configure the entire solution in the root (`config.yaml`).

### Local Development
You can also test it locally. We use UV for environment management. Here are linux / macos instructions for setting up a local environment, but here is the [official guide](https://github.com/astral-sh/uv#installation).

```bash
brew install uv
```

```bash
uv venv --python=3.12.3
source .venv/bin/activate
uv pip install . # base dependencies
uv pip install '.[dev]' # development
```

## Authors

<scott.mckean@databricks.com>
<chris.chalcraft@databricks.com>
<volo.vragov@databricks.com>

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
|docling|Document parsing and export|MIT|https://github.com/docling/docling|

Docling has quite a few dependencies: python-bidi, pyclipper, mpmath, filetype, XlsxWriter, typing-extensions, tqdm, tifffile, tabulate, sympy, soupsieve, shellingham, Shapely, scipy, safetensors, rtree, rpds-py, regex, pyyaml, python-dotenv, pypdfium2, pygments, pyflakes, pillow, opencv-python-headless, ninja, networkx, mdurl, MarkupSafe, marko, lxml, lazy-loader, jsonref, fsspec, filelock, et-xmlfile, dill, deepsearch-glm, click, attrs, annotated-types, referencing, python-pptx, python-docx, pydantic-core, openpyxl, multiprocess, mpire, markdown-it-py, jsonlines, jinja2, imageio, huggingface_hub, beautifulsoup4, autoflake, torch, tokenizers, scikit-image, rich, pydantic, jsonschema-specifications, docling-parse, typer, transformers, torchvision, semchunk, pydantic-settings, jsonschema, easyocr, docling-ibm-models, docling-core, docling

# Security Policy

## Reporting a Vulnerability

Please email bugbounty@databricks.com to report any security vulnerabilities. We will acknowledge receipt of your vulnerability and strive to send you regular updates about our progress. If you're curious about the status of your disclosure please feel free to email us again. If you want to encrypt your disclosure email, you can use [this PGP key](https://keybase.io/arikfr/key.asc).
