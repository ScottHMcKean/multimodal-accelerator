# Databricks notebook source
# MAGIC %md
# MAGIC # Convert (Optimized)
# MAGIC This module processes our bronze documents with key performance fixes:
# MAGIC - Pre-loaded models in actor constructors
# MAGIC - Non-blocking result processing
# MAGIC - Better error handling
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

from mlflow.models import ModelConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ModelConfig(development_config="config.yaml")
CATALOG = config.get("data").get("catalog")
SCHEMA = config.get("data").get("schema")
RAW_DOCS_VOL = config.get("data").get("raw_docs_vol")
PROCESSED_DOCS_VOL = config.get("data").get("processed_docs_vol")
OVERWRITE = config.get("data").get("overwrite")

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
from ray.util.spark import setup_ray_cluster

setup_ray_cluster(
    min_worker_nodes=4,
    max_worker_nodes=4,
    num_cpus_head_node=4,  # Use half our driver for processes too
    num_cpus_worker_node=8,  # Use all worker CPUs
    num_gpus_worker_node=0,
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can set some environment variables and pip things for the cluster

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from maud.utils import get_token

w = WorkspaceClient()
workspace_url = w.config.host
token = get_token(w)
ray.init(runtime_env={"env_vars": {"TOKEN": token, "WORKSPACE_URL": workspace_url}})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray MAUD Docling Converter (Optimized)
# MAGIC The MAUD acclerator extends Docling in order to process the tables, pages, and figures as well as the document hierarchy. We setup the whole converter pipeline within a single object with model pre-loading for better performance.

# COMMAND ----------

from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from pathlib import Path
from maud.document.converters import MAUDPipelineOptions, MAUDConverter, MAUDPipeline
from openai import OpenAI
import pandas as pd
import ray
import time
from typing import Optional
import warnings
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Suppress PyTorch pin_memory warnings
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.",
)
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.",
)

output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}")
output_dir.mkdir(parents=True, exist_ok=True)


@ray.remote(num_cpus=2, num_gpus=0, max_task_retries=3, max_restarts=2)
class DocumentProcessor:
    def __init__(self, workspace_url: str, token: str):
        """Pre-load models and initialize clients in constructor"""
        self.workspace_url = workspace_url
        self.token = token

        # Pre-initialize OpenAI client
        self.llm_client = OpenAI(
            api_key=token,
            base_url=f"{workspace_url}/serving-endpoints",
        )

        # Pre-configure MAUD options
        self.maud_options = MAUDPipelineOptions(
            llm_client=self.llm_client,
            llm_model="databricks-claude-3-7-sonnet",
            max_tokens=200,
            clf_client=self.llm_client,
            clf_model="dummy_clf",
            do_page_description=True,
            do_picture_description=True,
            generate_page_images=True,
            generate_picture_images=True,
            generate_table_images=True,
        )

        # Pre-load models by creating pipeline
        try:
            self.pipeline = MAUDPipeline(self.maud_options)
            logger.info("Models pre-loaded successfully")
        except Exception as e:
            logger.warning(f"Model pre-loading failed: {e}")
            self.pipeline = None

    def process_file(self, file_path: str) -> Optional[list]:
        """Process a single file"""
        try:
            output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}")

            converter = MAUDConverter(
                input_path=file_path,
                output_dir=output_dir,
                llm_client=self.llm_client,
                llm_model=self.maud_options.llm_model,
                max_tokens=self.maud_options.max_tokens,
                overwrite=OVERWRITE,
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=MAUDPipeline,
                        pipeline_options=self.maud_options,
                    )
                },
            )

            converter.convert()
            converter.save_document()
            chunks = converter.chunk()

            logger.info(f"Successfully processed {file_path}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return None


# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Processing with Dynamic Scaling

# COMMAND ----------

from pathlib import Path

max_files = 4
file_paths = sorted(
    Path(f"/Volumes/{CATALOG}/{SCHEMA}/{RAW_DOCS_VOL}").rglob("*.pdf"),
    key=lambda p: p.stat().st_size
)[:max_files]
total_files = len(file_paths)

# COMMAND ----------

# Get all PDF files
logger.info(f"Found {total_files} PDF files to process")

# Create actors with dynamic scaling
available_cpus = ray.available_resources().get("CPU", 8)
num_actors = max(2, min(total_files, int(available_cpus // 2)))

logger.info(f"Creating {num_actors} actors")

actors = [
    DocumentProcessor.options(max_restarts=2).remote(workspace_url, token)
    for _ in range(num_actors)
]

# Submit all tasks
futures = []
for i, file_path in enumerate(file_paths):
    actor_idx = i % num_actors
    future = actors[actor_idx].process_file.remote(str(file_path))
    futures.append(future)

logger.info(f"Submitted {len(futures)} processing tasks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Non-blocking Result Collection
# MAGIC This chunk executes the document parsing with non-blocking result collection - watch it run in the Ray Dashboard!

# COMMAND ----------

from ray.exceptions import RayTaskError
import time

results = []
completed = 0
failed = 0

# Process results as they complete (non-blocking)
while futures:
    # Wait for at least one task to complete
    ready, futures = ray.wait(futures, num_returns=1, timeout=30)

    # Process completed tasks
    for future in ready:
        try:
            result = ray.get(future)
            if result is not None:
                results.append(result)
                completed += 1
            else:
                failed += 1
        except RayTaskError as e:
            logger.error(f"Task failed: {e}")
            failed += 1
            results.append(None)

    logger.info(f"Progress: {completed + failed}/{total_files} files processed")

logger.info(f"Completed: {completed}, Failed: {failed}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save chunks to Delta
# MAGIC We now take the list of all the chunks we processed (this could be done asyncronously as well) and save into a delta table.

# COMMAND ----------

from itertools import chain
import pandas as pd

# Filter out None results and flatten
valid_results = [res for res in results if res is not None]
if valid_results:
    chunks_flat = list(chain.from_iterable(valid_results))

    logger.info(f"Creating DataFrame from {len(chunks_flat)} chunks")
    chunk_df = pd.DataFrame(chunks_flat)

    # Optimize data types
    chunk_df.input_hash = chunk_df.input_hash.astype(str)

    # Show sample results
    if "has_table" in chunk_df.columns:
        table_chunks = chunk_df.query("has_table == True")
        logger.info(f"Found {len(table_chunks)} table chunks")
        display(table_chunks.head(5) if len(table_chunks) > 0 else pd.DataFrame())

    # Save results
    output_path = f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}/chunks.parquet"
    chunk_df.to_parquet(output_path, index=False)

    logger.info(f"Saved {len(chunk_df)} chunks to {output_path}")

else:
    logger.warning("No valid results to save")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup

# COMMAND ----------

# Cleanup actors
for actor in actors:
    try:
        ray.kill(actor)
    except Exception:
        pass

# Shutdown Ray cluster
ray.shutdown()
logger.info("Ray cluster shutdown complete")

# COMMAND ----------


