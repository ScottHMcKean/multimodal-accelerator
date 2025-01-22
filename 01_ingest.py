# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest
# MAGIC
# MAGIC This notebook mocks the ingestion pipeline for our proof of concept. Every organisation uses a different configuration for ingestion, but we can abstract this into two main tasks:
# MAGIC
# MAGIC 1. Ingest documents from storage systems into cloud storage
# MAGIC 2. Replicate or mount this cloud storage into Unity Catalog Volumes
# MAGIC
# MAGIC The goal here is simply to land the documents and capture their lineage. The Extract Notebook covers the document processing.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC Our ingestion is driven off a table of URLs. This could also be easily done via URIs from blob storage (e.g. adfss) or volume paths. We use a default table in the `fixtures` folder to drive our loads, saving it as a delta table after we are done processing.

# COMMAND ----------

import pandas as pd
doc_df = pd.read_csv("./fixtures/maintenance_documents.csv")
doc_paths = spark.createDataFrame(doc_df)
display(doc_paths)

# COMMAND ----------

# MAGIC %md
# MAGIC We now use that driver table to download all the documents. We use a medallion architecture in volumes as well, landing in a bronze folder, processing into a silver folder, and serving from a gold folder.

# COMMAND ----------

#TODO: Add example and router for loading blob and/or volumes
import requests
from pathlib import Path
import pyspark.sql.functions as F
import pyspark.sql.types as T

SAVE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}"

@udf(T.BooleanType())
def download_file(title, url):
    response = requests.get(url)
    file_path = Path(SAVE_PATH) / f"{title}.pdf"
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return True

# Download all files in the dataframe
doc_paths = doc_paths.withColumn(
  'downloaded', download_file(F.col('document_name'), F.col('document_url'))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC We will also leverage delta tables to scale our work and leverage workers to reduce document processing time downstream.

# COMMAND ----------

@udf(T.BinaryType())
def load_url_to_binary(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        return None
      
# Download all files in the dataframe
doc_paths = doc_paths.withColumn(
  'bytes', load_url_to_binary(F.col('document_url'))
  )

# Save the DataFrame to a Delta table
( 
  doc_paths
  .write.format("delta")
  .mode("overwrite")
  .option("mergeSchema", "true")
  .saveAsTable(f"{CATALOG}.{SCHEMA}.{BRONZE_PATH}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC We now have our driver table with all the bytes for our document processing, as well as a copy in volumes to mock our cloud storage system. 

# COMMAND ----------

display(doc_paths)

# COMMAND ----------


