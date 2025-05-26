# Databricks notebook source
# MAGIC %md
# MAGIC # Convert (AI_PARSE)
# MAGIC This module processes our raw documents using AI_PARSE. It is the most involved and time consuming of the modules.
# MAGIC
# MAGIC The AI_PARSE notebook is a work in progress. Our goal will be to take the AI_PARSE output and generate the same table as our Docling / Ray but using AI_PARSE.

# COMMAND ----------

path = "/Volumes/shm/osc/raw_docs/12/Capital Group Prospectus.pdf"

# COMMAND ----------

import pyspark.sql.functions as F
df = spark.read.format("binaryFile").load(path)
df.display()

# COMMAND ----------

from pyspark.sql import functions as F

df = (
    spark.read.format("binaryFile").load(path)
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
