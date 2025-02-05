# Databricks notebook source
# MAGIC %md ##Chatmodel deployment 
# MAGIC
# MAGIC Supports batch (invoke) and streaming (stream) inference at the LangGraph event level.
# MAGIC

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md ####Define ChatModel

# COMMAND ----------

# MAGIC %%writefile chat_model.py
# MAGIC from random import randint
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ChatModel
# MAGIC from mlflow.types.llm import (ChatMessage, 
# MAGIC                               ChatCompletionResponse, 
# MAGIC                               ChatChoice, 
# MAGIC                               ChatChoiceDelta,
# MAGIC                               ChatChunkChoice,
# MAGIC                               ChatCompletionChunk)
# MAGIC from graph import load_graph
# MAGIC
# MAGIC
# MAGIC class GraphChatModel(ChatModel):
# MAGIC   """
# MAGIC   A mlflow ChatModel that invokes or streams a LangGraph
# MAGIC   workflow. 
# MAGIC
# MAGIC   The LangGraph output dictionary is parsed into mlflow chat model outputs
# MAGIC   that vary depending on whether the graph is invoked or streamed.
# MAGIC
# MAGIC   MLflow autologging for Langchain must be explicitly turned on. It is not on
# MAGIC   by default as is the case when logging a langchain model flavor.
# MAGIC   """
# MAGIC   def __init__(self):
# MAGIC     self.app = load_graph()
# MAGIC     mlflow.langchain.autolog()
# MAGIC
# MAGIC
# MAGIC   def format_chat_response(self, answer, message_history=None, stream=False):
# MAGIC     """
# MAGIC     Reformat the LangGraph dictionary output into mlflow chat model types.
# MAGIC     Streaming output requires the ChatCompletionChunk type; batch (invoke)
# MAGIC     output requires the ChatCompletionResponse type.
# MAGIC
# MAGIC     The models answer to the users question is returned. The messages history, 
# MAGIC     if it exists, is returned as a custom output within the chat message type.
# MAGIC     """
# MAGIC     if stream:
# MAGIC       chat_completion_response = ChatCompletionChunk(
# MAGIC                                   choices=[ChatChunkChoice(delta=ChatChoiceDelta(
# MAGIC                                     role = 'assistant',
# MAGIC                                     content=answer
# MAGIC                                   )
# MAGIC                                   )]
# MAGIC                               ) 
# MAGIC
# MAGIC     else:
# MAGIC       chat_completion_response = ChatCompletionResponse(
# MAGIC                                   choices=[ChatChoice(message=ChatMessage(role="assistant", 
# MAGIC                                                       content=answer))]
# MAGIC                               )
# MAGIC
# MAGIC     if message_history:
# MAGIC       chat_completion_response.custom_outputs = {"message_history": message_history}
# MAGIC
# MAGIC     return chat_completion_response
# MAGIC   
# MAGIC
# MAGIC   def predict_stream(self, context, messages, params=None):
# MAGIC     """
# MAGIC     NOTE: This method is not supported by Databricks model serving yet. 
# MAGIC     Stream the application on the input messages. Selectively choose the output 
# MAGIC     from graph events (node executions) to return to the user. This is necessary 
# MAGIC     when the model is executed using the 'stream' method rather than the 'invoke' 
# MAGIC     method. Each event in the stream represent the output of a LangGraph node. 
# MAGIC     The event is a dictionary where the key is the node's name and the value is the 
# MAGIC     nodes output dictionary, which conforms to the graph's state dictionary.
# MAGIC
# MAGIC     The output from nodes that invoke the model are return to the user. The
# MAGIC     last event returned from the stream will also include the message history
# MAGIC     as a custom output within the mlflow chat message type.
# MAGIC     """
# MAGIC     messages = {"messages": [message.to_dict() for message in messages]}
# MAGIC     for event in self.app.stream(messages):
# MAGIC
# MAGIC       if 'generation_with_history' in event:
# MAGIC         rewritten_question = event['generation_with_history']['generated_question'][-1]['content']
# MAGIC         rewritten_question_with_context = f"""To incorporate context from our conversation, I've rewritten your question as:
# MAGIC        
# MAGIC         '{rewritten_question}'
# MAGIC         """
# MAGIC         yield self.format_chat_response(rewritten_question_with_context, stream=True)
# MAGIC    
# MAGIC
# MAGIC       if 'generation_no_history' in event:
# MAGIC           message_history = event['generation_no_history']['messages']
# MAGIC           answer = message_history[-1]['content']
# MAGIC           yield self.format_chat_response(answer, message_history=message_history, stream=True)
# MAGIC
# MAGIC
# MAGIC   def predict(self, context, messages, params=None):
# MAGIC     """
# MAGIC     Apply the invoke(batch) method to the LangGraph application. The entire graph will
# MAGIC     execute and the completed state dictionary will be returned by the model.
# MAGIC
# MAGIC     Receives a list of messages in mlflow ChatMessage format. Convert the messages to a list
# MAGIC     of dictionaries before passing them to the application.
# MAGIC
# MAGIC     Elements from the dictionary are retrieved and reformatted into the expected MLflow 
# MAGIC     chat model type and are then returned.
# MAGIC     """
# MAGIC     messages = {"messages": [message.to_dict() for message in messages]}
# MAGIC     generation = self.app.invoke(messages)
# MAGIC     message_history = generation['messages']
# MAGIC     answer = message_history[-1]['content']
# MAGIC     return self.format_chat_response(answer, message_history=message_history)
# MAGIC
# MAGIC mlflow.models.set_model(GraphChatModel())

# COMMAND ----------

# MAGIC %md #### View graph

# COMMAND ----------

from IPython.display import Image
from chat_model import GraphChatModel

model = GraphChatModel()

display(Image(model.app.get_graph().draw_mermaid_png(output_file_path="graph.png")))

# COMMAND ----------

# MAGIC %md #### Test inference

# COMMAND ----------

from typing import List
from data.input_examples import input_example, input_examples
from nodes.utils import convert_to_chat_request, print_generation_and_history

input_examples_chat_format = []
for example in input_examples:
  input_examples_chat_format.append(convert_to_chat_request(example['messages']))

# COMMAND ----------

# MAGIC %md Batch

# COMMAND ----------

batch_generations = []
for example in input_examples_chat_format:
  generation = model.predict(None, example)
  batch_generations.append(generation)

# COMMAND ----------

print_generation_and_history(batch_generations, 0)

# COMMAND ----------

# MAGIC %md Streaming

# COMMAND ----------

stream_generations = []
for example in input_examples_chat_format:
  events = []
  for event in model.predict_stream(None, example):
    events.append(event)
  stream_generations.append(events)

# COMMAND ----------

print_generation_and_history(stream_generations, 2, streaming=True)

# COMMAND ----------

# MAGIC %md #### Log model to MLflow

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from data.input_examples import input_example, input_examples
from config_utils import load_config

retriever_config = load_config("retriever")
retriever_schema = retriever_config['schema']
vector_search_index_name = retriever_config['vector_search_index']

model_config = load_config("model")
model_name = model_config.get('name')

mlflow_config = load_config("mlflow")
experiment_location = mlflow_config["experiment_location"]
mlflow.set_experiment(experiment_location)
uc_model = mlflow_config["uc_model"]

mlflow.models.set_retriever_schema(
    primary_key=retriever_schema.get("primary_key"),
    text_column=retriever_schema.get("chunk_text"),
    doc_uri=retriever_schema.get("document_uri")  
)

with mlflow.start_run(run_name="graph_rag_pyfunc"):

  model_info = mlflow.pyfunc.log_model(
                  python_model = "chat_model.py",
                  streamable=True,
                  model_config="config.yaml",
                  artifact_path="graph",
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

# MAGIC %md ### Validate model inference

# COMMAND ----------

from mlflow.models import validate_serving_input
from mlflow.models.utils import load_serving_example

serving_payload = load_serving_example(model_uri)

validate_serving_input(model_uri, serving_payload)

# COMMAND ----------

# MAGIC %md #### Load mlflow model and test inference types.

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md Invoke

# COMMAND ----------

from pprint import pprint

predictions_invoke = loaded_model.predict(eval(serving_payload))
pprint(predictions_invoke)

# COMMAND ----------

# MAGIC %md Stream

# COMMAND ----------

streaming_results = []
for event in loaded_model.predict_stream(eval(serving_payload)):
  streaming_results.append(event)

# COMMAND ----------

print(f"{streaming_results[0]}\n")
print(f"{streaming_results[1]}\n")

# COMMAND ----------

# MAGIC %md ###Log as UC Model

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_info = mlflow.register_model(model_uri, 
                                   name = uc_model,
                                   tags={"model_type": "pyfunc",
                                         "streaming": True})

# COMMAND ----------

# MAGIC %md ### Deploy as Agent

# COMMAND ----------

from databricks.agents import deploy
from config_utils import load_config

mlflow_config = load_config("mlflow")
uc_model = mlflow_config.get('uc_model')

deployment_info = deploy(model_name=uc_model, 
                         model_version=model_info.version)

# COMMAND ----------

# MAGIC %md ### Query the endpoint

# COMMAND ----------

from pprint import pprint 
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")
endpoint_name = f"agents_{uc_model.replace('.', '-')}"

# COMMAND ----------

# MAGIC %md Invoke

# COMMAND ----------

result = deploy_client.predict(endpoint=deployment_info.endpoint_name, inputs=eval(serving_payload))
pprint(result)

# COMMAND ----------

for example in input_examples:
  result = deploy_client.predict(endpoint=deployment_info.endpoint_name, inputs=example)
pprint(result)
