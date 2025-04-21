# Databricks notebook source
# MAGIC %md
# MAGIC # Convert (Ray Parallel Version)
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

llm_model = "databricks-claude-3-7-sonnet"
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
    llm_model="databricks-claude-3-7-sonnet",
    max_tokens=200,
    clf_client=llm_client,
    clf_model='dummy_clf',
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Documents
# MAGIC We now use the extended MAUDConverter to do detailed document analysis

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *
import pandas as pd

# Define schema for output chunks
chunk_schema = StructType([
    StructField("doc_id", StringType()),
    StructField("chunk_text", StringType()),
    StructField("metadata", MapType(StringType(), StringType()))
])

@pandas_udf(chunk_schema)
def process_file_paths(file_paths: pd.Series) -> pd.DataFrame:
    """Processes file paths in parallel using Spark executors"""
    all_chunks = []
    
    # These need to be set per executor
    output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}")
    llm_client = initialize_llm_client()  # Implement your client initialization
    
    for file_path in file_paths:
        file_name = Path(file_path).name
        start_time = time.time()
        
        converter = MAUDConverter(
            input_path=file_path,
            output_dir=output_dir,
            llm_client=llm_client,
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
        
        converter.convert()
        converter.save_document()
        chunks = converter.chunk()
        
        # Convert chunks to pandas DataFrame format
        for chunk in chunks:
            all_chunks.append({
                "doc_id": file_name,
                "chunk_text": chunk.text,
                "metadata": chunk.metadata
            })
        
        print(f"Processed {file_name} in {time.time() - start_time:.1f}s")
    
    return pd.DataFrame(all_chunks)

# Parallel execution with Spark
processed_chunks_df = (documents_df
    .select("saved_file_path")
    .limit(2)  # Remove for full dataset
    .repartition(8, "saved_file_path")  # Adjust based on cluster size
    .mapInPandas(process_file_paths, schema=chunk_schema)
)

# Collect results if needed (avoid for large datasets)
all_chunks = [row.asDict() for row in processed_chunks_df.collect()]

# COMMAND ----------

from pathlib import Path
import time


documents_df = spark.table(f"{CATALOG}.{SCHEMA}.documents")

all_chunks = []
for row in documents_df.select("saved_file_path").collect()[0:2]:
    start_time = time.time()
    file_path = Path(row["saved_file_path"])
    file_name = file_path.name
    print(file_name)

    output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}")

    converter = MAUDConverter(
    input_path=file_path,
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

display(chunk_df.query("chunk_type == 'page'"))

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
