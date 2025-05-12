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

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC Our ingestion is driven off a table of URLs. This could also be easily done via URIs from blob storage (e.g. adfss) or volume paths. We use a default table in the `fixtures` folder to drive our loads, saving it as a delta table after we are done processing.

# COMMAND ----------

import pandas as pd
doc_df = pd.read_csv("./fixtures/forge_reports.csv")
doc_paths = spark.createDataFrame(doc_df)
display(doc_paths)

# COMMAND ----------

# MAGIC %md
# MAGIC We now use that driver table to download all the documents. We use a medallion architecture in volumes as well, landing in a bronze folder, processing into a silver folder, and serving from a gold folder.

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


