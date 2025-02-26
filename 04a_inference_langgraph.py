# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC This module bring everything together. We take our vector store, a foundation model, and implementation code and deploy a serving endpoint that can be used for multimodal retrieval.
# MAGIC
# MAGIC This subsection (04a) implements the inference flow using LangGraph

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
implementation_path = root_dir / "implementations" / "agents" / "langgraph"
mlflow_config = mlflow.models.ModelConfig(
    development_config=implementation_path / "config.yaml"
)
maud_config = parse_config(mlflow_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain
# MAGIC Setup our components and nodes

# COMMAND ----------

maud_config

# COMMAND ----------

# API Interfaces
from maud.agent.retrievers import get_vector_retriever
from databricks_langchain import ChatDatabricks

retriever = get_vector_retriever(maud_config)
model = ChatDatabricks(endpoint=maud_config.model.endpoint_name)

# Nodes
from maud.agent.states import get_state
from maud.agent.nodes import (
    make_query_vector_database_node,
    make_context_generation_node,
)

state = get_state(maud_config)
retriever_node = make_query_vector_database_node(retriever, maud_config)
context_generation_node = make_context_generation_node(model, maud_config)

# COMMAND ----------

# MAGIC %md
# MAGIC Setup the Graph

# COMMAND ----------

# Graph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from maud.agent.utils import graph_state_to_chat_type

workflow = StateGraph(state)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("generate_w_context", context_generation_node)
workflow.add_node("query_rewrite", rewrite_query)
workflow.add_edge(START, "query_rewrite")
workflow.add_edge("query_rewrite", "retrieve")
workflow.add_edge("retrieve", "generate_w_context")
workflow.add_edge("generate_w_context", END)
app = workflow.compile()

chain = app | RunnableLambda(graph_state_to_chat_type)

# COMMAND ----------

# MAGIC %md
# MAGIC Test our LangGraph Agent

# COMMAND ----------

input_example = {
    "messages": [
        {"role": "human", "content": "What is the factor of safety for fabric straps?"}
    ]
}

with mlflow.start_run():
    mlflow.langchain.autolog()
    result = chain.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log The Model
# MAGIC This is where the deployment magic happens. This may seem a little involved, but there is a lot of magic happening:
# MAGIC
# MAGIC - We set a retriever schema for MLFLow so that it can trace properly
# MAGIC - We set a well defined signature so that MLFLow knows that we can use the agent evaluation framework
# MAGIC - We provide a list of resources to allow flow through authentication
# MAGIC - We get a list of packages that matches of development package versions 

# COMMAND ----------

# Setup tracking and registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Setup experiment
mlflow.set_experiment(f"/Users/{USERNAME}/multimodal-langgraph")

# Setup retriever schema
mlflow.models.set_retriever_schema(
    primary_key=maud_config.retriever.mapping.primary_key,
    text_column=maud_config.retriever.mapping.chunk_text,
    doc_uri=maud_config.retriever.mapping.document_uri,
)

# Signature
from mlflow.models import ModelSignature
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA

signature = ModelSignature(
    inputs=CHAT_MODEL_INPUT_SCHEMA, outputs=CHAT_MODEL_OUTPUT_SCHEMA
)

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

# MAGIC %md
# MAGIC With all that in place, we make our Unity Catalog enabled MLFLow logging call. This does a couple things:
# MAGIC
# MAGIC - Uses code as the model to avoid serialization issues
# MAGIC - Passes in the 'maud' directory for our custom code
# MAGIC - Registers the model in Unity Catalog
# MAGIC - Provides an input example and signatures
# MAGIC - Provides the pass through resources

# COMMAND ----------

# Log the model
with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=str(implementation_path / "agent.py"),
        model_config=str(implementation_path / "config.yaml"),
        pip_requirements=packages,
        artifact_path="agent",
        code_paths=['maud'],
        registered_model_name=maud_config.agent.uc_model_name,
        input_example=input_example,
        signature=signature,
        resources=databricks_resources,
    )

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test the reloaded model before deploying it to ensure the inference works.

# COMMAND ----------

reloaded = mlflow.langchain.load_model(
    f"models:/{maud_config.agent.uc_model_name}/{logged_agent_info.registered_model_version}"
)
result = reloaded.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy
# MAGIC Now we deploy the model using the Mosaic AI Agents Framework. This provides some nice convenience features out of the box:
# MAGIC - A review app
# MAGIC - Integration with playground
# MAGIC - Inference tables & monitoring
# MAGIC - Versioned deployments

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from databricks import agents

client = get_deploy_client("databricks")

deployment_info = agents.deploy(
    maud_config.agent.uc_model_name,
    logged_agent_info.registered_model_version,
    scale_to_zero=True,
)
