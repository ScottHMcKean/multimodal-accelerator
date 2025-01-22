# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC In this module we take our 'feature store' (our vector search index), leverage a foundation model with Databricks, and do retrieval on our multimodal documents.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import os
import mlflow
from operator import itemgetter
from mlflow.models import ModelConfig

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from databricks_langchain.vectorstores import DatabricksVectorSearch
from databricks_langchain.chat_models import ChatDatabricks

mlflow.langchain.autolog()
config = ModelConfig(development_config='src/maud/configs/agent_config.yaml')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundation Model
# MAGIC
# MAGIC Let's first setup a foundation model using the ChatDatabricks interface

# COMMAND ----------

llm_config = config.get("llm")

llm = ChatDatabricks(
    endpoint=llm_config.get("endpoint_name"), 
    max_tokens=llm_config.get("max_tokens"), 
    temperature=llm_config.get("temperature")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retriever
# MAGIC
# MAGIC Let's setup the retriever

# COMMAND ----------

vs_config = config.get("vector_search")

vector_search = DatabricksVectorSearch(
    endpoint=vs_config.get("endpoint_name"),
    index_name=vs_config.get("index_name"),
    columns=[
        vs_config.get('primary_key'), 
        vs_config.get('text_column'), 
        vs_config.get('doc_uri')
    ] + vs_config.get("other_columns", [])
)
    
vs_retriever = vector_search.as_retriever(
    search_type=vs_config.get("search_type"), 
    search_kwargs={
        'k': vs_config.get('num_results'), 
        "score_threshold": vs_config.get('score_threshold'),
        'query_type': vs_config.get('query_type')
        }
)

mlflow.models.set_retriever_schema(
    primary_key= vs_config.get('primary_key'),
    text_column= vs_config.get('text_column'),
    doc_uri= vs_config.get('doc_uri'),
    name="vs_index",
)

# COMMAND ----------

vs_retriever.invoke('Reference Parts List')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval Chain
# MAGIC Here we use a retrieval chain to pull our documents

# COMMAND ----------

prompt = PromptTemplate(
    template=config.get("system_prompt"), 
    input_variables=["context", "input"]
    )

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vs_retriever, combine_docs_chain)

mlflow.models.set_model(rag_chain)

# COMMAND ----------

results = rag_chain.invoke({"input":"What are dams that have failed?"})

# COMMAND ----------

results['context'][0]

# COMMAND ----------

from mlflow.models import infer_signature
input = {"input":"Should I use shear protection when towing an aircraft?"}
signature = infer_signature(input, rag_chain.invoke(input))

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)


with mlflow.start_run():
    # Set the registry URI to Unity Catalog if needed
    mlflow.set_registry_uri('databricks-uc')

    # Define the list of Databricks resources needed to serve the agent
    list_of_databricks_resources = [
        DatabricksServingEndpoint(endpoint_name=llm_config.get("endpoint_name")),
        DatabricksVectorSearchIndex(index_name=vs_config.get("index_name")),
    ]

    # Log the model in MLflow with the signature  
    logged_agent_info = mlflow.langchain.log_model(
        lc_model="src/maud/agents/agent.py",
        model_config='src/maud/configs/agent_config.yaml',
        artifact_path="model",
        pip_requirements=[
            "langchain==0.3.13",
            "langchain-community==0.3.13",
            "pydantic==2.10.4",
            "databricks-langchain==0.1.1"
        ],
        resources=list_of_databricks_resources, 
        signature=signature,
        registered_model_name="shm.multimodal.retrieval_agent",
        )

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Test the reloaded model before deploying

# COMMAND ----------

reloaded = mlflow.langchain.load_model("models:/shm.multimodal.retrieval_agent/2")
result = reloaded.invoke({"input":"Should I use shear protection when towing an aircraft?"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy 
# MAGIC Now we deploy the model

# COMMAND ----------

from mlflow.deployments import get_deploy_client
client = get_deploy_client("databricks")

# Deploy the model to serving
deploy_name = "maud-agent"
model_name = "shm.multimodal.retrieval_agent"
model_version = 2

endpoint = client.create_endpoint(
    name=deploy_name,
    config={
        "served_entities": [{
            "entity_name": model_name,
            "entity_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
        }
)
