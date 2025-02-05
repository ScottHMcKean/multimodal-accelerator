# Databricks notebook source
# MAGIC %md ##Langchain deployment
# MAGIC
# MAGIC Supports batch (invoke) inference
# MAGIC

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md #### Define Langchain chain

# COMMAND ----------

# MAGIC %%writefile chain.py
# MAGIC import mlflow
# MAGIC from langchain_core.runnables import RunnableGenerator, RunnableLambda
# MAGIC from graph import load_graph
# MAGIC from nodes.utils import graph_state_to_chat_type
# MAGIC
# MAGIC
# MAGIC def load_chain():
# MAGIC   app = load_graph()
# MAGIC   chain = app | RunnableLambda(graph_state_to_chat_type)
# MAGIC   return chain
# MAGIC
# MAGIC mlflow.models.set_model(load_chain())

# COMMAND ----------

# MAGIC %md #### View graph

# COMMAND ----------

from IPython.display import Image
from chain import load_chain

app = load_chain()
display(Image(app.get_graph().draw_mermaid_png(output_file_path="graph.png")))

# COMMAND ----------

# MAGIC %md #### Test inference

# COMMAND ----------

from data.input_examples import input_example, input_examples

example_generations = []
for example in input_examples:
  invoke_event = app.invoke(example)
  example_generations.append(invoke_event)

# COMMAND ----------

# MAGIC %md The customer outputs look good; the historical messages are as expected.

# COMMAND ----------

example_generations[0]

# COMMAND ----------

# MAGIC %md ###Log to MLflow and Deploy

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.models import ModelSignature
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA
from data.input_examples import input_example, input_examples
from config_utils import load_config

retriever_config = load_config("retriever")
retriever_schema = retriever_config['schema']
vector_search_index_name = retriever_config['vector_search_index']

model_config = load_config("model")
model_name = model_config['name']

mlflow_config = load_config("mlflow")
experiment_location = mlflow_config["experiment_location"]
mlflow.set_experiment(experiment_location)
uc_model = mlflow_config["uc_model"]

mlflow.models.set_retriever_schema(
    primary_key=retriever_schema.get("primary_key"),
    text_column=retriever_schema.get("chunk_text"),
    doc_uri=retriever_schema.get("document_uri")  
)

explicit_signature = ModelSignature(inputs=CHAT_MODEL_INPUT_SCHEMA, outputs=CHAT_MODEL_OUTPUT_SCHEMA)

with mlflow.start_run(run_name="graph_rag_chain"):
  model_info = mlflow.langchain.log_model(
                  lc_model="chain.py",
                  streamable=False,
                  model_config="config.yaml",
                  artifact_path="graph",
                  signature=explicit_signature,
                  input_example=input_example,
                  
                  code_paths = [
                    'nodes',
                    'prompts',
                    'resources',
                    'state.py',
                    'graph.py',
                    'config_utils.py'
                    ],
                  
                  resources = [
                    DatabricksVectorSearchIndex(index_name=vector_search_index_name),
                    DatabricksServingEndpoint(endpoint_name=model_name)
                    ],
                  pip_requirements = "requirements.txt"
               )
  
  mlflow.log_artifact("graph.png")

  model_uri = model_info.model_uri

  loaded_app = mlflow.pyfunc.load_model(model_uri)
  for example in input_examples:
    loaded_app.predict(example)

print(model_uri)

# COMMAND ----------

# MAGIC %md Validate model

# COMMAND ----------

from mlflow.models import validate_serving_input
from mlflow.models.utils import load_serving_example

serving_payload = load_serving_example(model_uri)

validate_serving_input(model_uri, serving_payload)

# COMMAND ----------

# MAGIC %md Log to model registry

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_info = mlflow.register_model(model_uri, 
                                   name=uc_model,
                                   tags={"model_type": "chain",
                                         "streaming": False})

# COMMAND ----------

# MAGIC %md Deploy as Agent

# COMMAND ----------

from databricks.agents import deploy
from config_utils import load_config

mlflow_config = load_config("mlflow")
uc_model = mlflow_config.get('uc_model')

deployment_info = deploy(model_name=uc_model, 
                         model_version=model_info.version)

# COMMAND ----------

# MAGIC %md ### Deployment feedback
# MAGIC  - Review App
# MAGIC  - API call

# COMMAND ----------

# MAGIC %md Query the endpoint using the Python API. 
# MAGIC
# MAGIC These look good

# COMMAND ----------

from pprint import pprint 
from mlflow.deployments import get_deploy_client
from data.input_examples import input_examples

deploy_client = get_deploy_client("databricks")

endpoint_name = f"agents_{'main.default.mlc_langgraph_model'.replace('.', '-')}"

endpoint_results = []
for example in input_examples:
  result = deploy_client.predict(endpoint=endpoint_name, inputs=example)
  endpoint_results.append(result)

# COMMAND ----------

endpoint_results[2]

# COMMAND ----------

# MAGIC %md Review inference tables
