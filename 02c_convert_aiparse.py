# Databricks notebook source
# MAGIC %md
# MAGIC # Convert (AI_PARSE)
# MAGIC This module processes our raw documents using AI_PARSE. It is the most involved and time consuming of the modules.
# MAGIC
# MAGIC The AI_PARSE notebook is a work in progress. For now it is just a demonstration to show how AI_PARSE works. The ultimate goal will be to take the AI_PARSE output and generate the same table as our Docling / Ray for compatibility.
# MAGIC
# MAGIC This notebook works with Serverless, but you must use Environment version 2.

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %restart_python

# COMMAND ----------

from mlflow.models import ModelConfig
import pyspark.sql.functions as F
import pandas as pd

config = ModelConfig(development_config="config.yaml")
CATALOG = config.get("data").get("catalog")
SCHEMA = config.get("data").get("schema")
RAW_DOCS_VOL = config.get("data").get("raw_docs_vol")
doc_paths = spark.table(f"{CATALOG}.{SCHEMA}.documents")

# COMMAND ----------

# MAGIC %md
# MAGIC We use the smallest file for a quick demonstration

# COMMAND ----------

from pathlib import Path
all_paths = doc_paths.select("saved_file_path").collect()
paths_list = [row["saved_file_path"] for row in all_paths]
sizes = [(path, Path(path).stat().st_size) for path in paths_list]
smallest_file_path = min(sizes, key=lambda x: x[1])[0]

# COMMAND ----------

file_path = doc_paths.select("saved_file_path").first()["saved_file_path"]
df = spark.read.format("binaryFile").load(smallest_file_path)
df.display()

# COMMAND ----------

df = (
    spark.read.format("binaryFile").load(smallest_file_path)
  .select(
    F.col("path"),
    F.expr("ai_parse(content)").alias("parsed"))
  .withColumn(
    "parsed_json",
    F.parse_json(F.col("parsed").cast("string")))
  .select(
    F.col("path"),
    F.expr("parsed_json:document").alias("document"),
    F.expr("parsed_json:pages").alias("pages"),
    F.expr("parsed_json:elements").alias("elements"),
    F.expr("parsed_json:_corrupted_data").alias("_corrupted_data"))
)

display(df)

# COMMAND ----------

from pyspark.sql.functions import from_json, explode, col
from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, StringType, MapType

# Parse the VARIANT/JSON column to array of structs
element_schema = StructType([
    StructField("id", IntegerType()),
    StructField("page_indices", ArrayType(IntegerType())),
    StructField("representation", StructType([
        StructField("markdown", StringType()),
        StructField("text", StringType())
    ])),
    StructField("schema", StringType()),
    StructField("summary", StringType()),
    StructField("title", StringType()),
    StructField("type", StringType())
])

df = df.withColumn("elements_array", from_json(col("elements").cast("string"), ArrayType(element_schema)))

# Now explode and extract fields
df_exploded = df.withColumn("element", explode(col("elements_array")))

df_flat = df_exploded.select(
    col("path"),
    col("element.id").alias("id"),
    col("element.page_indices").alias("page_indices"),
    col("element.representation.markdown").alias("markdown"),
    col("element.representation.text").alias("text"),
    col("element.schema").alias("schema"),
    col("element.summary").alias("summary"),
    col("element.title").alias("title"),
    col("element.type").alias("type")
).drop("element", "elements_array")

# COMMAND ----------

df_flat.display()
