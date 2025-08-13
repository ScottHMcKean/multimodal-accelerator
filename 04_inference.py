# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC This module bring everything together. We take our vector store, a foundation model, and implementation code and deploy an agent that can be used for multimodal retrieval. This notebook used LangGraph as the main deployment framework (along with lots of MLFLow) and has been tested on serverless.

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Parse our config using pydantic types and validation to standardize productionized workflow

# COMMAND ----------

# load config as both an mlflow object and a pydantic class
import mlflow
from src.agent.config import parse_config

mlflow_config = mlflow.models.ModelConfig(development_config="config.yaml")
maud_config = parse_config(mlflow_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain
# MAGIC Setup our components and nodes

# COMMAND ----------

# API Interfaces
from src.agent.retrievers import get_vector_retriever
from databricks_langchain import ChatDatabricks

retriever = get_vector_retriever(maud_config)
model = ChatDatabricks(endpoint=maud_config.model.endpoint_name)

# Nodes
from src.agent.states import get_state
from src.agent.nodes import (
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
from src.agent.utils import graph_state_to_chat_type

workflow = StateGraph(state)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("generate_w_context", context_generation_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate_w_context")
workflow.add_edge("generate_w_context", END)
app = workflow.compile()

# COMMAND ----------

input_example = {"messages": [{"role": "user", "content": "How do I add a new layer?"}]}

# COMMAND ----------

# MAGIC %md
# MAGIC We can now predict using the compiled graph

# COMMAND ----------

result = app.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use the graph in streaming mode

# COMMAND ----------

for msg in app.stream(input_example, stream_mode="updates"):
    print(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC Test our LangGraph Agent

# COMMAND ----------

from importlib import reload
import agent

reload(agent)
from agent import MAUDAgent

AGENT = MAUDAgent(app)
AGENT.predict(input_example)

# COMMAND ----------

for msg in AGENT.predict_stream(input_example):
    print(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC

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
mlflow.set_experiment(maud_config.agent.experiment_location)

# Setup retriever schema
mlflow.models.set_retriever_schema(
    primary_key=maud_config.retriever.primary_key,
    text_column=maud_config.retriever.text_column,
    doc_uri=maud_config.retriever.document_uri,
)

# Setup passthrough resources
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)

databricks_resources = [
    DatabricksServingEndpoint(endpoint_name=maud_config.model.endpoint_name),
    DatabricksVectorSearchIndex(
        index_name=f"{maud_config.data.uc_catalog}.{maud_config.data.uc_schema}.{maud_config.retriever.index_name}"
    ),
]

# Get dependencies
import tomllib

with open("pyproject.toml", "rb") as f:
    toml = tomllib.load(f)
dependencies = toml["project"]["dependencies"]

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
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        model_config="config.yaml",
        artifact_path="agent",
        code_paths=["src"],
        pip_requirements=dependencies,
        registered_model_name=maud_config.agent.uc_model_name,
        input_example=input_example,
        resources=databricks_resources,
    )

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

maud_config.retriever.chunk_template

# COMMAND ----------

import mlflow

# COMMAND ----------

# Log the prompts
prompt = mlflow.genai.register_prompt(
    name="shm.pid.retriever",
    template=maud_config.retriever.chunk_template,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test the reloaded model before deploying it to ensure the inference works.

# COMMAND ----------

reloaded = mlflow.pyfunc.load_model(
    f"models:/{maud_config.agent.uc_model_name}/{logged_agent_info.registered_model_version}"
)
result = reloaded.predict(input_example)

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
