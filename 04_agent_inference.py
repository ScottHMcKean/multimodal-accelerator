# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC In this module we take our 'feature store' (our vector search index), leverage a foundation model with Databricks, and do retrieval on our multimodal documents. We use the agent framework to do this

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import mlflow

DEV_CONFIG_FILE = 'src/maud/configs/agent_config.yaml'
AGENT_FILE = 'src/maud/agents/tool_agent.py'
config = mlflow.models.ModelConfig(development_config=DEV_CONFIG_FILE)

# COMMAND ----------

from maud.agents.tool_agent import FunctionCallingAgent
fc_agent = FunctionCallingAgent()
response = fc_agent.predict(messages=[
  {"role": "user", "content": "What is the factor of safety for strapping?"}
  ])

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionRequest, ChatCompletionResponse, StringResponse
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)

vs_config = config.get("vector_search")
llm_config = config.get("llm")

input_example = {
    "messages": [
        {"role": "user", "content": "What is the factor of safety for strapping?"}
    ]
}

with mlflow.start_run():
    # Set the registry URI to Unity Catalog if needed
    mlflow.set_registry_uri('databricks-uc')

    # Define the list of Databricks resources needed to serve the agent
    list_of_databricks_resources = [
        DatabricksServingEndpoint(endpoint_name=llm_config.get("endpoint_name")),
        DatabricksVectorSearchIndex(index_name=vs_config.get("index_name")),
    ]

    logged_agent_info = mlflow.pyfunc.log_model(
        python_model=AGENT_FILE,
        model_config=DEV_CONFIG_FILE,
        artifact_path='agent',
        pip_requirements=[
            "mlflow>=2.20.0",
            "backoff",
            "databricks-agents",
            "databricks-sdk[openai]",
            "databricks-vectorsearch"
        ],
        registered_model_name='shm.multimodal.tool_agent',
        input_example=input_example,
        resources=list_of_databricks_resources
    )

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Test the reloaded model before deploying

# COMMAND ----------

reloaded = mlflow.pyfunc.load_model(
  f"models:/shm.multimodal.tool_agent/{logged_agent_info.registered_model_version}"
  )
result = reloaded.predict(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy 
# MAGIC Now we deploy the model

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from databricks import agents

client = get_deploy_client("databricks")

# Deploy the model to serving
deploy_name = "maud-agent"
model_name = "shm.multimodal.tool_agent"
model_version = logged_agent_info.registered_model_version

deployment_info = agents.deploy(model_name, model_version)
