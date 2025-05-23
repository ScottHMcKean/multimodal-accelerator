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

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Ray
# MAGIC We want our document processing to be fast and work in batches, but not cost a fortune. So we use Ray to parallelize the work off a table
# MAGIC
# MAGIC One gotcha here is that the autoscaling doesn't work well with custom libraries and requirements. If you don't reserve and initialize all the nodes at once, you won't have the custom packages. There are workarounds for this, but the easiest solution is to reserve all the workers prior to running ray.init().
# MAGIC
# MAGIC You can see how Ray is working with the Ray dashboard below.

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

setup_ray_cluster(
    min_worker_nodes=6,
    max_worker_nodes=6,
    num_cpus_head_node=4,
    num_cpus_worker_node=8,
    num_gpus_worker_node=0,
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
    token = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )
else:
    token = workspace_client.config.token

ray.init(runtime_env={"env_vars": {"TOKEN": token, "WORKSPACE_URL": workspace_url}})

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

    def process_file(self, country="France"):
        self._init_client()

        # Use client with custom base_url
        response = self.llm_client.chat.completions.create(
            model="databricks-claude-3-7-sonnet",
            messages=[
                {"role": "user", "content": f"What is the capital of {country}?"}
            ],
        )
        return response.choices[0].message.content


# COMMAND ----------


countries = ["France", "Germany", "Japan", "Brazil"]

processor = CapitalProcessor.remote()
results = ray.get([processor.process_file.remote(country) for country in countries])
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

import ray


@ray.remote(num_cpus=2, num_gpus=0, max_task_retries=3)
class DocumentProcessor:
    def __init__(self):
        """Empty constructor - no non-serializable objects here"""
        pass

    def process_file(self, file_path: str, workspace_url: str, token: str):
        """All initialization happens within method execution"""
        # Worker-side path creation
        output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize client on worker
        llm_client = OpenAI(
            api_key=token,
            base_url=f"{workspace_url}/serving-endpoints",
        )

        maud_options = MAUDPipelineOptions(
            llm_client=llm_client,
            llm_model="databricks-claude-3-7-sonnet",
            max_tokens=200,
            clf_client=llm_client,  # Reuse same client
            clf_model="dummy_clf",
            describe_pages=True,
            describe_tables=True,
            describe_pictures=True,
        )

        converter = MAUDConverter(
            input_path=file_path,
            output_dir=output_dir,
            llm_client=maud_options.llm_client,
            llm_model=maud_options.llm_model,
            max_tokens=maud_options.max_tokens,
            overwrite=False,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=MAUDPipeline,
                    pipeline_options=MAUDPipelineOptions(),
                )
            },
        )

        converter.convert()
        converter.save_document()
        return converter.chunk()


# COMMAND ----------

ray.available_resources()

# COMMAND ----------

from pathlib import Path

file_paths = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{RAW_DOCS_VOL}").rglob("*.pdf")

# Manual actor sharding due to init process
num_actors = 12
actors = [DocumentProcessor.options(max_restarts=3).remote() for _ in range(num_actors)]

futures = [
    actors[i % num_actors].process_file.remote(path, workspace_url, token)
    for i, path in enumerate(file_paths)
]

# COMMAND ----------

from ray.exceptions import RayTaskError

results = []
for future in futures:
    try:
        results.append(ray.get(future))
    except RayTaskError as e:
        print(f"Task failed: {e}")
        results.append(None)  # or log error details

all_chunks = [res for res in results if res is not None]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save chunks to Delta
# MAGIC We now take the list of all the chunks we processed (this could be done asyncronously as well) and save into a delta table.

# COMMAND ----------

from itertools import chain

chunks_flat = list(chain.from_iterable(all_chunks))
chunk_df = pd.DataFrame(chunks_flat)

# COMMAND ----------

chunk_df.input_hash = chunk_df.input_hash.astype(str)
chunk_df.query("has_table == True").head(5)

# COMMAND ----------

ray.shutdown()

# COMMAND ----------

chunk_df.to_parquet(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}/chunks.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC You might need to run this on serverless if you've taken all the spark nodes

# COMMAND ----------

# import pandas as pd
# chunk_df = pd.read_parquet(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}/chunks.parquet")

# COMMAND ----------

# from pyspark.sql.functions import monotonically_increasing_id
# chunk_sp = spark.createDataFrame(chunk_df)
# chunk_sp = chunk_sp.withColumn("id", monotonically_increasing_id())
# chunk_sp.write.option("mergeSchema", "true").mode("overwrite").saveAsTable(f"{CATALOG}/{SCHEMA}.chunks")
# display(chunk_sp)
