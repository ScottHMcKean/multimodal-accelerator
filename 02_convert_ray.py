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
# MAGIC ## Setup Ray
# MAGIC We want our document processing to be fast and work in batches, but not cost a fortune. So we use Ray to parallelize the work off a table

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

setup_ray_cluster(
  min_worker_nodes=3,
  max_worker_nodes=3, 
  num_cpus_head_node=4,
  num_cpus_worker_node=4,
  num_gpus_worker_node=0
  )

# COMMAND ----------

# MAGIC %md
# MAGIC We can set some environment variables and pip things for the cluster

# COMMAND ----------

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

ray.init(
  runtime_env={
    "env_vars": {
      "TOKEN": token,
      "WORKSPACE_URL": workspace_url
    }
  }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray Tests
# MAGIC This is a basic test of running a ray process on a list of strings

# COMMAND ----------

from openai import OpenAI
import ray

@ray.remote
class CapitalProcessor:
    def __init__(self):
        """Client initialization happens on worker nodes"""
        self.workspace_url = os.environ["WORKSPACE_URL"]
        self.token = os.environ["TOKEN"]
        self.llm_client = None  # Deferred initialization
    
    def _init_client(self):
        """Initialize client with custom base_url"""
        if not self.llm_client:
            self.llm_client = OpenAI(
                api_key=self.token,
                base_url=f"{self.workspace_url}/serving-endpoints",
            )
    
    def process_file(self, country='France'):
        self._init_client()
        
        # Use client with custom base_url
        response = self.llm_client.chat.completions.create(
            model="databricks-claude-3-7-sonnet",
            messages=[{"role": "user", "content": f"What is the capital of {country}?"}]
        )
        return response.choices[0].message.content

# COMMAND ----------


countries = ["France", "Germany", "Japan", "Brazil"]

processor = CapitalProcessor.remote()
results = ray.get([
    processor.process_file.remote(country)
    for country in countries
])
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray MAUD Docling Converter
# MAGIC The MAUD acclerator extends Docling in order to process the tables, pages, and figures as well as the document hierarchy. We setup the whole converter pipeline within a single object

# COMMAND ----------

from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from maud.document.converters import MAUDPipelineOptions, MAUDConverter, MAUDPipeline
import pandas as pd

from openai import OpenAI
import ray

@ray.remote(num_cpus=2, num_gpus=0)
class DocumentProcessor:
    def __init__(self):
        """Client initialization happens on worker nodes"""
        self.workspace_url = os.environ["WORKSPACE_URL"]
        self.token = os.environ["TOKEN"]
        self.llm_client = None  # Deferred initialization
    
    def _init_client(self):
        """Initialize client with custom base_url"""
        if not self.llm_client:
            self.llm_client = OpenAI(
                api_key=self.token,
                base_url=f"{self.workspace_url}/serving-endpoints",
            )

        self.maud_pipeline_options = MAUDPipelineOptions(
            llm_client=self.llm_client,
            llm_model="databricks-claude-3-7-sonnet",
            max_tokens=200,
            clf_client=self.llm_client,
            clf_model='dummy_clf',
            describe_pages=False,
            describe_tables=False,
            describe_pictures=False
        )
    
    def process_file(self, file_path):
        output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}")
        self._init_client()
        
        # Use client with custom base_url
        self.converter = MAUDConverter(
            input_path=file_path,
            output_dir=output_dir,
            llm_client=self.maud_pipeline_options.llm_client,
            llm_model=self.maud_pipeline_options.llm_model,
            max_tokens=self.maud_pipeline_options.max_tokens,
            overwrite=False,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=MAUDPipeline,
                    pipeline_options=MAUDPipelineOptions(),
                )
                }
            )

        result = self.converter.convert()
        self.converter.save_document()
        doc_chunks = self.converter.chunk()
        return doc_chunks

# COMMAND ----------

from pathlib import Path
file_paths = sorted(Path('/Volumes/shm/multimodal/raw_docs/').glob('*'), key=lambda x: x.stat().st_size)[:6]

processor = DocumentProcessor.remote()
results = ray.get([processor.process_file.remote(path) for path in file_paths])
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Documents
# MAGIC We now use the extended MAUDConverter to do detailed document analysis

# COMMAND ----------

from pathlib import Path
import time


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
