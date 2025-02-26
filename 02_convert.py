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

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup LLM Client
# MAGIC We use our workspace client to configure both local and notebook execution for our LLM client for describing images. We keep costs low by using 4o-mini via the Mosaic AI model gateway

# COMMAND ----------

# Get LLM Client
from openai import OpenAI
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

workspace_client = WorkspaceClient()
workspace_url = workspace_client.config.host

# Check if running in Databricks
import os

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
else:
    token = workspace_client.config.token

llm_model = "shm_gpt_4o_mini"
llm_client = OpenAI(
    api_key=token,
    base_url=f"{workspace_url}/serving-endpoints",
)

# Test LLM Client
llm_client.chat.completions.create(
    model=llm_model,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MAUD Docling Converter
# MAGIC The MAUD acclerator extends Docling in order to process the tables, pages, and figures as well as the document hierarchy.

# COMMAND ----------

from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from maud.document.converters import MAUDPipelineOptions, MAUDConverter, MAUDPipeline
import pandas as pd

maud_pipeline_options = MAUDPipelineOptions(
    llm_client=llm_client,
    llm_model="shm_gpt_4o_mini",
    max_tokens=200,
    clf_client=llm_client,
    clf_model='dummy_clf',
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Documents
# MAGIC We now use the extended MAUDConverter to do detailed document analysis

# COMMAND ----------

from pathlib import Path
import time

# iterative on files from the bronze path
files = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}").glob("*")
types = [".xlsx", ".docx", ".pptx", ".pdf"]
filenames = [file.name for file in files if file.suffix in types]

all_chunks = []
for filename in filenames:
    start_time = time.time()
    print(filename)

    input_path = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}/{filename}")
    output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{SILVER_PATH}")

    converter = MAUDConverter(
    input_path=input_path,
    output_dir=output_dir,
    llm_client=maud_pipeline_options.llm_client,
    llm_model=maud_pipeline_options.llm_model,
    max_tokens=maud_pipeline_options.max_tokens,
    overwrite=False,
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=MAUDPipeline,
            pipeline_options=MAUDPipelineOptions(),
        )
        }
    )

    result = converter.convert()
    converter.save_document()
    doc_chunks = converter.chunk()
    all_chunks.extend(doc_chunks)
    print(f"time(s): {round(time.time() - start_time, 1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save chunks to Delta
# MAGIC We now take the list of all the chunks we processed (this could be done asyncronously as well) and save into a delta table.

# COMMAND ----------

import pandas as pd
chunk_df = pd.DataFrame(all_chunks)
chunk_df.input_hash = chunk_df.input_hash.astype(str)
chunk_df

# COMMAND ----------

from maud.document.chunkers import chunk_schema
from pyspark.sql.functions import monotonically_increasing_id

chunk_sp = spark.createDataFrame(chunk_df)
chunk_sp = chunk_sp.withColumn("id", monotonically_increasing_id())
chunk_sp.write.option("mergeSchema", "true").mode("overwrite").saveAsTable("shm.multimodal.processed_chunks")
display(chunk_sp)
