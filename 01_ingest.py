# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest
# MAGIC
# MAGIC This notebook mocks the ingestion pipeline for our proof of concept. Every organisation uses a different configuration for ingestion, but we can abstract this into two main tasks:
# MAGIC
# MAGIC 1. Ingest documents from storage systems into cloud storage
# MAGIC 2. Replicate or mount this cloud storage into Unity Catalog Volumes
# MAGIC
# MAGIC The goal here is simply to land the documents and capture their lineage via a clear save path. The `02_convert` Notebook covers the document processing.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Our ingestion is driven off a table of URLs. This could also be easily done via URIs from blob storage (e.g. adfss) or volume paths. We use a default table in the `assets` folder to drive our loads, saving it as a delta table after we are done processing.

# COMMAND ----------

from mlflow.models import ModelConfig

config = ModelConfig(development_config="config.yaml")
CATALOG = config.get("data").get("catalog")
SCHEMA = config.get("data").get("schema")
RAW_DOCS_VOL = config.get("data").get("raw_docs_vol")
PROCESSED_DOCS_VOL = config.get("data").get("processed_docs_vol")

import pandas as pd

doc_df = pd.read_csv("./assets/forge_reports.csv")
doc_paths = spark.createDataFrame(doc_df)
display(doc_paths)

# COMMAND ----------

# MAGIC %md
# MAGIC We now use that driver table to download all the documents. We use a medallion architecture in volumes as well, landing in a bronze folder, processing into a silver folder, and serving from a gold folder.

# COMMAND ----------

# Ensure volumes are ready
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    w.catalogs.create(name=CATALOG)
except:
    log.info(f"{CATALOG} catalog exists")

try:
    w.schemas.create(catalog_name=CATALOG, name=SCHEMA)
except:
    log.info(f"{SCHEMA} catalog exists")

for vol_name in [RAW_DOCS_VOL, PROCESSED_DOCS_VOL]:
    try:
        w.volumes.create(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            name=vol_name,
            volume_type=VolumeType.MANAGED,
        )
    except:
        log.info(f"{vol_name} volume exists")


# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
import requests
from maud.document.utils import sanitize_filename

RAW_DOC_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{RAW_DOCS_VOL}"


@udf(T.StringType())
def download_file(url):
    file_name = sanitize_filename(url)
    saved_file_path = f"{RAW_DOC_DIR}/{file_name}"
    response = requests.get(url)
    with open(saved_file_path, "wb") as file:
        file.write(response.content)
    return saved_file_path


# COMMAND ----------

# Download all files in the dataframe in parallel
doc_paths = doc_paths.withColumn(
    "saved_file_path", download_file(F.col("download_link"))
)

(
    doc_paths.write.format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.documents")
)

# COMMAND ----------

display(doc_paths)

# COMMAND ----------
