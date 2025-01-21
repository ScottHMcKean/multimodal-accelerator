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
from mlflow.models import ModelConfig

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import TransformChain
from langchain.chains import SequentialChain

from databricks_langchain.vectorstores import DatabricksVectorSearch
from databricks_langchain.chat_models import ChatDatabricks

mlflow.langchain.autolog()
config = ModelConfig(development_config='src/maud/config/default/agent_config.yaml')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundation Model
# MAGIC
# MAGIC Let's first setup a foundation model using the ChatDatabricks interface

# COMMAND ----------

llm_config = config.get("llm")

chat_model = ChatDatabricks(
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

vs_config.get("other_columns")

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
    template=config.get("retrieval_prompt"), 
    input_variables=["context", "question"]
    )

retrieval_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vs_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

mlflow.models.set_model(retrieval_chain)

# COMMAND ----------

result = retrieval_chain(
  {"query":"Should I use shear protection when towing an aircraft?"}
  )
print(result['result'])
result['source_documents']

# COMMAND ----------

[x.metadata['img_path'] for x in result['source_documents']]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Driver
# MAGIC Now we can deploy this agent as a serving endpoint to use within our interface

# COMMAND ----------

from mlflow.models import infer_signature
input = {"query":"Should I use shear protection when towing an aircraft?"}
signature = infer_signature(input, retrieval_chain(input))

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
        lc_model="src/maud/agent/agent.py",
        model_config='src/maud/config/default/agent_config.yaml',
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

reloaded = mlflow.langchain.load_model("models:/shm.multimodal.retrieval_agent/1")
result = reloaded.invoke({"query":"Should I use shear protection when towing an aircraft?"})

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
model_version = 1

endpoint = client.create_endpoint(
    name=f"{deploy_name}_{model_version}",
    config={
        "served_entities": [{
            "entity_name": model_name,
            "entity_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
        }
)

# COMMAND ----------


