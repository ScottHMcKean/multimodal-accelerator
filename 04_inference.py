# Databricks notebook source
# MAGIC %md
# MAGIC # Inference
# MAGIC
# MAGIC In this module we take our 'feature store' (our vector search index), leverage a foundation model with Databricks, and do retrieval on our multimodal documents.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig

from databricks_langchain.vectorstores import DatabricksVectorSearch
from databricks_langchain.chat_models import ChatDatabricks

mlflow.langchain.autolog()
config = ModelConfig(development_config="maud/config/agent_config.yaml")

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
    temperature=llm_config.get("temperature"),
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
    columns=[vs_config.get("primary_key"), vs_config.get("doc_uri")]
    + vs_config.get("other_columns", []),
)

retriever = vector_search.as_retriever(
    search_type=vs_config.get("search_type"),
    search_kwargs={
        "k": vs_config.get("num_results"),
        "score_threshold": vs_config.get("score_threshold"),
        "query_type": vs_config.get("query_type"),
    },
)

mlflow.models.set_retriever_schema(
    primary_key=vs_config.get("primary_key"),
    text_column=vs_config.get("text_column"),
    doc_uri=vs_config.get("doc_uri"),
    name="vs_index",
)

# COMMAND ----------

documents = retriever.invoke("Factor of safety for fabric materials")

# COMMAND ----------

type(documents[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval Chain
# MAGIC Here we use a retrieval chain to pull our documents

# COMMAND ----------

from typing import List, TypedDict, Annotated, Dict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph

# COMMAND ----------


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: dict


# COMMAND ----------

system_prompt = config.get("system_prompt")

# COMMAND ----------


def retrieve(state: State):
    last_message = state["messages"][-1]
    docs = retriever.invoke(last_message.content)

    merged_doc_dict = [doc.to_json() for doc in docs]

    context = "\n\n".join(
        f"ID: {int(doc.metadata['id'])}\nContent: {doc.page_content}\nSource: {doc.metadata['img_path']}"
        for doc in docs
    )

    return {
        "messages": [SystemMessage(content=system_prompt + context)],
        "documents": merged_doc_dict,
    }


# COMMAND ----------


def generate(state: State):
    messages = state["messages"]
    system_message = [m for m in messages if m.type == "system"]
    messages_for_llm = system_message + [m for m in messages if m.type == "human"]
    response = llm.invoke(messages_for_llm)
    return {"messages": [response]}


# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import json


def parse_messages_docs(state) -> str:
    last_message = state["messages"][-1]
    parser = StrOutputParser()
    parsed_output = parser.parse(last_message.content)

    documents: dict = state.get("documents", {})

    # Create a JSON object
    output = {"response": parsed_output, "documents": documents}

    # Convert the dictionary to a JSON string
    return json.dumps(output, indent=2)


# COMMAND ----------

graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")

graph = graph.compile()
chain = graph | RunnableLambda(parse_messages_docs)

# COMMAND ----------

input_example = {
    "messages": [
        {"role": "human", "content": "What is the factor of safety for fabric straps?"}
    ]
}
result = chain.invoke(input_example)

# COMMAND ----------

from mlflow.models import infer_signature

signature = infer_signature(input_example, result)

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    StringResponse,
)
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)

with mlflow.start_run():
    # Set the registry URI to Unity Catalog if needed
    mlflow.set_registry_uri("databricks-uc")

    # Define the list of Databricks resources needed to serve the agent
    list_of_databricks_resources = [
        DatabricksServingEndpoint(endpoint_name=llm_config.get("endpoint_name")),
        DatabricksVectorSearchIndex(index_name=vs_config.get("index_name")),
    ]

    logged_agent_info = mlflow.langchain.log_model(
        lc_model="maud/agents/maud_agent.py",
        model_config="maud/configs/agent_config.yaml",
        pip_requirements=[
            "langgraph>=0.2.62",
            "pydantic",
            "databricks_langchain",
            "databricks-vectorsearch",
        ],
        artifact_path="agent",
        registered_model_name="shm.multimodal.retrieval_agent",
        input_example=input_example,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(),
        ),
        resources=list_of_databricks_resources,
    )

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Test the reloaded model before deploying

# COMMAND ----------

reloaded = mlflow.langchain.load_model("models:/shm.multimodal.retrieval_agent/15")
result = reloaded.invoke(input_example)

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
model_name = "shm.multimodal.retrieval_agent"
model_version = 15

deployment_info = agents.deploy(model_name, model_version)

# This is the alternative, but the agents framework is better for authentication passthrough
# endpoint = client.create_endpoint(
#     name=deploy_name,
#     config={
#         "served_entities": [{
#             "entity_name": model_name,
#             "entity_version": model_version,
#             "workload_size": "Small",
#             "scale_to_zero_enabled": True
#         }]
#         }
# )
