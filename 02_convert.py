# Databricks notebook source
# MAGIC %md
# MAGIC # Convert
# MAGIC
# MAGIC This module processes our bronze documents. It is the most involved of the modules but we leverage Docling as a framework to abstract away the layout analysis of a document. Docling takes a list of files in the volumes, parallelized over workers.
# MAGIC
# MAGIC In order to make this result useful downstream, we need to do three things:
# MAGIC
# MAGIC - Export the result to a reloadable json format
# MAGIC - Save and export the images
# MAGIC - Save and export the tables
# MAGIC
# MAGIC In order to get the exports, we can modify the defaults and generate page, picture, and table images. This really slows down the parsing pipeline, but is essential for our user interface to reload and serve everything.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC This section runs through the parsing and export of a single document. This takes a while since we are OCRing, extracting images, and analyzing the layout of each document. Docling provides a nice framework for exporting these documents as markdown files, both with linked and embedded images. This makes the downstream take of summarizing and converting images to text much easier, where we can even replace the images with a list of symbols references, description, caption etc.

# COMMAND ----------

# Get LLM Client
from openai import OpenAI
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

workspace_client = WorkspaceClient()
workspace_url = workspace_client.config.host
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
# MAGIC ### Setup Docling Converter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from maud.document.converters import MAUDPipelineOptions, MAUDConverter, MAUDPipeline

maud_pipeline_options = MAUDPipelineOptions(
    llm_client=llm_client,
    llm_model="shm_gpt_4o_mini",
    max_tokens=200,
    clf_client=llm_client,
    clf_model="dummy_clf",
)

# COMMAND ----------
from pathlib import Path
import time

# iterative on files from the bronze path
files = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}").glob("*")
types = [".xlsx", ".docx", ".pptx", ".pdf"]
filenames = [file.name for file in files if file.suffix in types]

iter_results = {}
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
        overwrite=True,
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=MAUDPipeline,
                pipeline_options=maud_pipeline_options,
            )
        },
    )

# convert document and save it
document = converter.convert()
converter.save_result()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Document
# MAGIC Here we chunk the text, tables, pictures, and pages into consistent chunks for vector search.
# COMMAND ----------

import pandas as pd
from maud.document.chunkers import chunk_schema

chunks = converter.chunk()
chunk_df = pd.DataFrame(chunks)
chunk_sp = spark.createDataFrame(chunk_df, chunk_schema)
chunk_sp.write.mode("overwrite").saveAsTable("shm.multimodal.processed_chunks")
display(chunk_sp)
