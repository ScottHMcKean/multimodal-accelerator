# Multimodal Document Processing

This project follows a simple four-step flow:
1. Parse (`1_parse.ipynb`)
2. Serving (`2_serving.ipynb`)
3. Agentify (`3_agentify.ipynb`)
4. Evaluate (`4_evaluate.ipynb`)

## Quick start
1. Drop documents into the input volume (`/Volumes/<catalog>/<schema>/<input_volume>`) or the local input path (`local_input_path` in `config.yaml`).
2. Optional: use the commented cell in `1_parse.ipynb` to copy the repo samples from `examples/` into the input directory.
3. Run `1_parse.ipynb`
4. Run `2_serving.ipynb`
5. Run `3_agentify.ipynb`
6. Run `4_evaluate.ipynb`

## Databricks Asset Bundle
This repo includes a Databricks Asset Bundle for deploying the notebook jobs.

Common commands:
- `databricks bundle deploy -t dev`
- `databricks bundle run docling_parse -t dev`
- `databricks bundle run docling_serving -t dev`
- `databricks bundle run docling_agentify -t dev`
- `databricks bundle run docling_evaluate -t dev`

## Serving endpoint tool contract
Deployment uses an agent tool that calls the model serving endpoint. It accepts input paths and returns parsed output paths.

Request format (dataframe_split):
- columns: `["file_path", "output_root", "options"]`
- data: rows of file paths and a shared output root

Response format:
- `predictions`: list of objects with `status` and `output_path`

## Authors

<scott.mckean@databricks.com>
<chris.chalcraft@databricks.com>
<volo.vragov@databricks.com>

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.

## License

&copy; 2026 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library | description | license | source |
|---|---|---|---|
| docling | Document parsing and export | MIT | https://github.com/docling-project/docling |
| mlflow | ML lifecycle management | Apache Software License | https://github.com/mlflow/mlflow |
| databricks-sdk | Databricks SDK for Python | Apache Software License | https://github.com/databricks/databricks-sdk-py |
| databricks-connect | Local Spark connectivity | Other/Proprietary License | https://pypi.org/project/databricks-connect/ |
| databricks-vectorsearch | Vector search client | UNKNOWN | https://pypi.org/project/databricks-vectorsearch/ |
| databricks-ai-bridge | Databricks AI bridge | Databricks License | https://pypi.org/project/databricks-ai-bridge/ |
| onnxruntime | ONNX Runtime | MIT License | https://github.com/microsoft/onnxruntime |
| pandas | Data analysis | BSD License | https://github.com/pandas-dev/pandas |
| pillow | Imaging library | MIT-CMU License | https://github.com/python-pillow/Pillow |
| pymssql | MS SQL Server driver | LGPL-2.1 | https://github.com/pymssql/pymssql |
| requests | HTTP library | Apache Software License | https://github.com/psf/requests |
| ipykernel | Jupyter kernel | BSD 3-Clause License | https://github.com/ipython/ipykernel |

# Security Policy

## Reporting a Vulnerability

Please email bugbounty@databricks.com to report any security vulnerabilities. We will acknowledge receipt of your vulnerability and strive to send you regular updates about our progress. If you're curious about the status of your disclosure please feel free to email us again. If you want to encrypt your disclosure email, you can use [this PGP key](https://keybase.io/arikfr/key.asc).
