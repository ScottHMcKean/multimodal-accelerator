# Databricks notebook source
# MAGIC %md
# MAGIC # Convert
# MAGIC This module processes our bronze documents. It is the most involved and time consuming of the modules. We leverage the Docling framework to abstract away the layout analysis of a document.
# MAGIC
# MAGIC In order to make this useful downstream for multimodal vector search, we need three things:
# MAGIC
# MAGIC - Exported tables, images, and pages
# MAGIC - A reloadable and cachable conversion
# MAGIC - Vector search ready text chunks that also incorporate tables and figures
# MAGIC
# MAGIC This notebook is designed to be used with a classic cluster using ML Runtime 15.4 LTS, with CPU compute. It was tested with a STANDARD_e8_v3 with 6 workers in Azure. The goal is to reduce cost as much as possible by using cheap CPU workers with high utilization across workers.

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from pathlib import Path

pdf_input_path = Path("tests/data/wiring_bonding.pdf")
output_dir = Path("data/processed")  # Define output directory

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from openai import OpenAI
from maud.utils import get_token

w = WorkspaceClient()
workspace_url = w.config.host
token = get_token(w)

llm_model = "databricks-claude-3-7-sonnet"
llm_client = OpenAI(
    api_key=token,
    base_url=f"{workspace_url}/serving-endpoints",
)


# COMMAND ----------

import warnings

warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray MAUD Docling Converter
# MAGIC The MAUD acclerator extends Docling in order to process the tables, pages, and figures as well as the document hierarchy. We setup the whole converter pipeline within a single object

# COMMAND ----------

from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from maud.document.converters import MAUDPipelineOptions, MAUDConverter, MAUDPipeline
import pandas as pd

maud_options = MAUDPipelineOptions(
    llm_client=llm_client,
    llm_model="databricks-claude-3-7-sonnet",
    max_tokens=200,
    clf_client=None,
    clf_model="dummy_clf",
    do_picture_description=True,
    do_page_description=True,
    generate_page_images=True,
    generate_picture_images=True,
    generate_table_images=True,
)

# COMMAND ----------

converter = MAUDConverter(
    input_path=pdf_input_path,
    output_dir=output_dir,
    llm_client=maud_options.llm_client,
    llm_model=maud_options.llm_model,
    max_tokens=maud_options.max_tokens,
    overwrite=True,
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=MAUDPipeline,
            pipeline_options=maud_options,
        )
    },
)

document = converter.convert()

# COMMAND ----------

document.model_dump().keys()

# COMMAND ----------

converter.save_document()
chunks = converter.chunk()

# COMMAND ----------

pd.DataFrame(chunks)

# COMMAND ----------
