# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC This module bring everything together. We take our vector store, a foundation model, and implementation code and deploy a serving endpoint that can be used for multimodal retrieval.
# MAGIC
# MAGIC This subsection (04c) implements the inference flow using an OpenAI tool calling agent.
# MAGIC
# MAGIC This is currently a work in progress

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Parse our config using pydantic types and validation to standardize productionized workflow

# COMMAND ----------

# load config
from pathlib import Path
import mlflow
from maud.agent.config import parse_config
import os

root_dir = Path(os.getcwd())
implementation_path = root_dir / 'implementations' / 'agents' / 'openai' 
mlflow_config = mlflow.models.ModelConfig(
  development_config=implementation_path / 'config.yaml'
  )
maud_config = parse_config(mlflow_config)

# COMMAND ----------

from implementations.agents.openai.tool_agent import FunctionCallingAgent
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

input_example = {
    "messages": [
        {"role": "user", "content": "What is the factor of safety for strapping?"}
    ]
}

# Set the registry URI to Unity Catalog if needed
    mlflow.set_registry_uri('databricks-uc')

list_of_databricks_resources = [
        DatabricksServingEndpoint(endpoint_name=llm_config.get("endpoint_name")),
        DatabricksVectorSearchIndex(index_name=vs_config.get("index_name")),
    ]

with mlflow.start_run():
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
