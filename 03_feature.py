# Databricks notebook source
# MAGIC %md
# MAGIC # Featurize
# MAGIC With Generative AI and RAG specifically, most of the 'featurization' of our data is simply preparing a vector search index. This notebook takes the processed chunks (which contain tables, images, pages, and text chunks)

# COMMAND ----------

catalog = "shm"
schema = "osc"
chunk_table = "chunks"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index
# MAGIC Now that we have a single table for vector search, let's load it into Databricks Vector Search. There is more we can do here, but for now we simply want to

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE shm.multimodal.processed_chunks
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC We pull the configuration from the agent configuration. The most important part of the vector index setup is the columns we are going to sync - in order to do filtering and advanced retrieval, we need to make sure those columns are available.

# COMMAND ----------

from mlflow.models import ModelConfig
config = ModelConfig(development_config="implementations/agents/langgraph/config.yaml")

vs_config = config.get("retriever")
vs_endpoint = vs_config.get("endpoint_name")
vs_index_name = vs_config.get("index_name")
vs_source_table = vs_config.get("source_table")
vs_cols_to_sync = [
  'filename',
  'input_hash',
  'pages',
  'chunk_type',
  'image_path',
  'text',
  'enriched_text',
  'id']

# COMMAND ----------

# MAGIC %md
# MAGIC If the index exists already, we will run a sync on the table. If not, we will use the SDK to create the index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from agent.retrievers import index_exists
client = VectorSearchClient()
try:
    index = client.get_index(vs_endpoint, vs_index_name)
    index.sync()
except:
    index = client.create_delta_sync_index(
        endpoint_name=vs_endpoint,
        source_table_name=vs_source_table,
        index_name=vs_index_name,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name="databricks-gte-large-en",
        columns_to_sync=vs_cols_to_sync,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search As Tool

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM vector_search(
# MAGIC   index=>'shm.osc.prospectus_index',
# MAGIC   query_text=>"Blackrock investments",
# MAGIC   num_results=>5,
# MAGIC   query_type=>"hybrid"
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION shm.osc.prospectus_search(
# MAGIC   query STRING COMMENT 'A query that should resemble a section of a prospectus'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   matching_descriptions STRING
# MAGIC )
# MAGIC COMMENT 'Returns the top 5 entries of financial prospectus. When using this tool, you should rewrite the user query into something that would appear on a financial prospectus. It also includes the heading, filename and pages where the chunk appeared'
# MAGIC RETURN
# MAGIC SELECT CONCAT(
# MAGIC   'Filename: ', filename,
# MAGIC   '\n Pages: ', CAST(pages AS string),
# MAGIC   '\n Type: ', chunk_type,
# MAGIC   '\n Text: ', enriched_text
# MAGIC ) as result
# MAGIC FROM vector_search(
# MAGIC   index=>'shm.osc.prospectus_index',
# MAGIC   query_text=>query,
# MAGIC   num_results=>5,
# MAGIC   query_type=>'hybrid'
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM shm.osc.prospectus_search('Blackrock Investments')
