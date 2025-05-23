# Databricks notebook source
# MAGIC %md
# MAGIC # Convert
# MAGIC This module processes our bronze documents. It is the most involved and time consuming of the modules. In this excersize we use the AI_PARSE() functionality.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC %restart_python


# COMMAND ----------

path = "/Volumes/shm/osc/raw_docs/12/Capital Group Prospectus.pdf"

# COMMAND ----------

import pyspark.sql.functions as F

df = spark.read.format("binaryFile").load(path)

df.display()

# COMMAND ----------

df = (
    spark.read.format("binaryFile")
    .load(path)
    .select(F.col("path"), F.expr("parse_unstructured(content)").alias("parsed"))
    .withColumn("parsed_json", F.parse_json(F.col("parsed").cast("string")))
    .select(
        F.col("path"),
        F.expr("parsed_json:document").alias("document"),
        F.expr("parsed_json:pages").alias("pages"),
        F.expr("parsed_json:elements").alias("elements"),
        F.expr("parsed_json:_corrupted_data").alias("_corrupted_data"),
    )
)
display(df)
