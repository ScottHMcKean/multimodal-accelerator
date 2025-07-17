# Databricks notebook source
# MAGIC %md
# MAGIC # Featurize
# MAGIC With Generative AI and RAG specifically, most of the 'featurization' of our data is simply preparing a vector search index. This notebook takes the processed chunks (which contain tables, images, pages, and text chunks). It has been tested with Serverless.

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
sys.path.append(".")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move converted index to a table

# COMMAND ----------

# MAGIC %md
# MAGIC Because our conversion process overtook spark, we reload the cached chunks.parquet file and save it to a table

# COMMAND ----------

from mlflow.models import ModelConfig
import pandas as pd

config = ModelConfig(development_config="config.yaml")

# data
CATALOG = config.get("data").get("catalog")
SCHEMA = config.get("data").get("schema")
CHUNKS_TABLE_NAME = config.get("data").get("chunks_table_name")
PROCESSED_DOCS_VOL = config.get("data").get("processed_docs_vol")

# retriever
VS_ENDPOINT = config.get("retriever").get("endpoint_name")
INDEX_NAME = config.get("retriever").get("index_name")
EMBEDDING_MODEL = config.get("retriever").get("embedding_model")
NUM_RESULTS = config.get("retriever").get("num_results",5)
QUERY_TYPE = config.get("retriever").get("search_type","hybrid")
KEY = config.get("retriever").get("primary_key")
TEXT_COL = config.get("retriever").get("text_column")

# COMMAND ----------

chunk_df = pd.read_parquet(f"/Volumes/{CATALOG}/{SCHEMA}/{PROCESSED_DOCS_VOL}/chunks.parquet")

# COMMAND ----------

chunk_df

# COMMAND ----------

def cast_to_str_array(arr):
    if arr is None:
        return []
    # Convert all elements to string, skip None values
    return [str(x) for x in arr if x is not None]

# COMMAND ----------

chunk_df_no_tables = chunk_df.drop(columns=['tables'])

# COMMAND ----------

chunk_sp = spark.createDataFrame(chunk_df_no_tables)

# COMMAND ----------

chunk_sp

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
chunk_sp = spark.createDataFrame(chunk_df_no_tables)
chunk_sp = chunk_sp.withColumn("id", monotonically_increasing_id())
# chunk_sp.write.option("mergeSchema", "true").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.chunks")
display(chunk_sp.limit(5))

# COMMAND ----------

chunk_sp.write.option("mergeSchema", "true").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.chunks")

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.{CHUNKS_TABLE_NAME}
    ADD COLUMNS (tables ARRAY<STRING>)
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE devanshu_pandey.multimodal.chunks
# MAGIC     ADD COLUMNS (tables ARRAY<STRING>)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from devanshu_pandey.multimodal.chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index
# MAGIC Now that we have a single table for vector search, let's load it into Databricks Vector Search. There is more we can do here, but for now we simply want to

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.{CHUNKS_TABLE_NAME} 
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC We pull the configuration from the agent configuration. The most important part of the vector index setup is the columns we are going to sync - in order to do filtering and advanced retrieval, we need to make sure those columns are available. If the index exists already, we will run a sync on the table. If not, we will use the SDK to create the index

# COMMAND ----------

def index_exists(client, vs_endpoint, index_name):
    try:
        client.get_index(vs_endpoint, index_name)
        return True
    except Exception as e:
        if "IndexNotFoundException" in str(e):
            return False
        else:
            raise e

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()
try:
    index = client.get_index(VS_ENDPOINT, INDEX_NAME)
    index.sync()
except:
    index = client.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        source_table_name=f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE_NAME}",
        index_name=f"{CATALOG}.{SCHEMA}.{INDEX_NAME}",
        pipeline_type="TRIGGERED",
        primary_key=KEY,
        embedding_source_column=TEXT_COL,
        embedding_model_endpoint_name=EMBEDDING_MODEL
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search As Tool

# COMMAND ----------

query = "How do I create a new layer?"

spark.sql(f"""
SELECT *
FROM vector_search(
  index=>'{CATALOG}.{SCHEMA}.{INDEX_NAME}',
  query_text=>'{query}',
  num_results=>{NUM_RESULTS},
  query_type=>'{QUERY_TYPE}'
)
""").display()

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.search(
  query STRING COMMENT 'A query that should resemble a section of a technical document and have at least 20 words in it'
)
RETURNS TABLE (
  matching_descriptions STRING
)
COMMENT 'A vector search of technical documents. It also includes the heading, filename and pages where the chunk appeared'
RETURN
SELECT CONCAT(
  'Filename: ', filename,
  '\n Pages: ', CAST(pages AS string),
  '\n Type: ', chunk_type,
  '\n Text: ', enriched_text
) as result
FROM vector_search(
  index=>'{CATALOG}.{SCHEMA}.{INDEX_NAME}',
  query_text=>query,
  num_results=>{NUM_RESULTS},
  query_type=>'{QUERY_TYPE}'
)
""")

# COMMAND ----------

spark.sql(f"""
  SELECT * 
  FROM {CATALOG}.{SCHEMA}.search('{query}')
""").display()
