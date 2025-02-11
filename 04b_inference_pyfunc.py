# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC This submodule continues the work from 04a, but instead deploys langgraph using MLFLow ChatModel and a pyfunc process.

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
implementation_path = root_dir / "implementations" / "agents" / "pyfunc"
mlflow_config = mlflow.models.ModelConfig(
    development_config=implementation_path / "config.yaml"
)
maud_config = parse_config(mlflow_config)

# COMMAND ----------

from implementations.agents.pyfunc.agent import GraphChatModel

# COMMAND ----------

# MAGIC %md
# MAGIC Test a standard prediction

# COMMAND ----------

input_example = {
    "messages": [
        {"role": "human", "content": "What is the factor of safety for fabric straps?"}
    ]
}

with mlflow.start_run():
    model = GraphChatModel()
    model.predict(None, input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log
# MAGIC Log the agent using mlflow

# COMMAND ----------

# Setup tracking and registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Setup experiment
mlflow.set_experiment(f"/Users/{USERNAME}/multimodal-pyfunc")

# Setup retriever schema
mlflow.models.set_retriever_schema(
    primary_key=maud_config.retriever.mapping.primary_key,
    text_column=maud_config.retriever.mapping.chunk_text,
    doc_uri=maud_config.retriever.mapping.document_uri,
)

# Signature
from mlflow.models import ModelSignature
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA

# Setup passthrough resources
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)

databricks_resources = [
    DatabricksServingEndpoint(endpoint_name=maud_config.model.endpoint_name),
    DatabricksVectorSearchIndex(index_name=maud_config.retriever.index_name),
]

# Get packages from requirements to set standard environments
with open("requirements.txt", "r") as file:
    packages = file.readlines()
    package_list = [pkg.strip() for pkg in packages]

# COMMAND ----------

# Log the model
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model=str(implementation_path / "agent.py"),
        streamable=True,
        model_config=str(implementation_path / "config.yaml"),
        pip_requirements=packages,
        artifact_path="agent",
        code_paths=['maud'],
        registered_model_name=maud_config.agent.uc_model_name,
        input_example=input_example,
        resources=databricks_resources,
    )

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Test the reloaded model before deploying

# COMMAND ----------

reloaded = mlflow.pyfunc.load_model(
    f"models:/{maud_config.agent.uc_model_name}/{logged_agent_info.registered_model_version}"
)
result = reloaded.predict(input_example)

# COMMAND ----------

response_gen = reloaded.predict_stream(input_example)
next(response_gen)

# COMMAND ----------

next(response_gen)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy
# MAGIC Now we deploy the model

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from databricks import agents

client = get_deploy_client("databricks")

# Deploy the model to serving
deploy_name = "multimodal-pyfunc"

deployment_info = agents.deploy(
    maud_config.agent.uc_model_name,
    logged_agent_info.registered_model_version,
    scale_to_zero=True,
)
