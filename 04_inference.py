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
config = ModelConfig(development_config='config.yml')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundation Model
# MAGIC
# MAGIC Let's first setup a foundation model using the ChatDatabricks interface

# COMMAND ----------

llm_max_tokens = config.get("llm_max_tokens")
llm_temperature = config.get("llm_temperature")

chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct", 
    max_tokens=llm_max_tokens, 
    temperature=llm_temperature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retriever
# MAGIC
# MAGIC Let's setup the retriever

# COMMAND ----------

vs_endpoint = config.get("vs_endpoint")
vs_index_name = config.get("vs_index_name")
score_threshold = config.get('score_threshold')
num_results = config.get('num_results')

def get_retriever(persist_dir: str = None):
    vector_search = DatabricksVectorSearch(
        index_name=vs_index_name,
        columns=['id', 'filename', 'img_path', 'pages', 'text', 'type']
        )
    
    return vector_search.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={
            'k': num_results, 
            "score_threshold": score_threshold,
            'query_type': 'hybrid'
            }
    )

vs_retriever = get_retriever()

mlflow.models.set_retriever_schema(
    primary_key='id',
    text_column='content',
    doc_uri='img_path',
    name="vs_index",
)

# COMMAND ----------

vs_retriever.invoke('Reference Parts List')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval Chain
# MAGIC Here we use a retrieval chain to pull our documents

# COMMAND ----------

retrieval_template = config.get("retrieval_prompt")

prompt = PromptTemplate(
    template=retrieval_template, 
    input_variables=["context", "question"]
    )

def get_filtered_retriever(filters:dict = {}):
    retriever = get_retriever()
    retriever.search_kwargs["filters"] = filter
    return retriever

def retrieval_qa_chain(inputs):
    question = inputs["question"]
    filtered_retriever = get_retriever()
    
    # Define the RetrievalQA chain with the filtered retriever
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=filtered_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    # Run the retrieval chain
    output = retrieval_chain({"query": question})
    # Return the outputs
    return output

# COMMAND ----------

result = retrieval_qa_chain(
  {"question":"Should I use shear protection when towing an aircraft?"}
  )
result

# COMMAND ----------

result['result']

# COMMAND ----------

result['source_documents'][0]

# COMMAND ----------

retrieval_qa_chain({"question":"What parts are in the parts reference list?"})

# COMMAND ----------

output = retrieval_qa_chain({"question":"What bombardier planes is this equipment good for?"})
output['result']

# COMMAND ----------
# MAGIC %md
# MAGIC ## Agent
# MAGIC Now we can deploy this agent as a serving endpoint to use within our interface

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
        DatabricksServingEndpoint(endpoint_name="shm-gpt-4o-mini"),
        DatabricksVectorSearchIndex(index_name="shm-vs-index"),
    ]

    # Define the custom signature
    input_schema = Schema([
        ColSpec("string", "recipe"),
        ColSpec("long", "customer_count")
    ])
    output_schema = Schema([ColSpec("string")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Log the model in MLflow with the signature  
    logged_agent_info = mlflow.langchain.log_model(
        lc_model="./agent.py",
        artifact_path="model",
        pip_requirements=[
            "langchain==0.3.13",
            "langchain-community==0.3.13",
            "pydantic==2.10.4",
            "databricks-langchain==0.1.1"
        ],
        resources=list_of_databricks_resources, 
        signature=signature)

    uc_model_info = mlflow.register_model(
        model_uri=logged_agent_info.model_uri, 
        name="shm.default.mlflow_example")

    print(f"Model logged and registered with URI: {logged_agent_info.model_uri}")