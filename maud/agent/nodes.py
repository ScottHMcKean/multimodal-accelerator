from typing import Iterator, Dict, List, Any
import mlflow
from langchain_core.runnables import RunnableLambda
from state import GraphState, StreamState
from prompts.templates import prompt_no_history, prompt_with_history
from resources.retriever import vector_search_retriever, format_documents
from resources.model import model
from nodes.utils import (format_generation_user,
                         format_generation_assistant,
                         choose_prompt_question)
from config_utils import load_config

config = load_config("langgraph")
support_streaming: bool = config.get("streaming")
graph_state = StreamState if support_streaming  else GraphState 

def load_generation_no_history():
  """
  Load the propery implementation for the generation_no_history
  node depending on whether streaming or batch inference
  is specified in config.yaml
  """
  if graph_state:
    return generation_no_history_stream
  else:
    return generation_no_history


def contains_chat_history(state: graph_state) -> str:
  """
  Determine if conversation history exists. Ask the model to 
  rephrase the user's current question such that it incorporates
  the conversation history.
  """
  history = state['messages']
  return "contains_history" if len(history) > 1 else "no_history"


def query_vector_database(state: graph_state) -> Dict[str, Any]:
  """
  Retrieve and format the documents from the vector index.
  """
  question = choose_prompt_question(state)[0]

  documents = vector_search_retriever.invoke(question['content'])

  formatted_documents = format_documents(documents)

  return {"context": formatted_documents}


def generation_no_history(state: graph_state) -> Dict[str, Any]:
  """
  Inject the users's question, which may have been previously rephrased
  by the model if there was message history, and the context retrieved 
  from the vector index into the prompt.
  """
  question = choose_prompt_question(state)

  chain = prompt_no_history | model | RunnableLambda(format_generation_assistant)

  generation = chain.invoke({"context": state["context"],
                             "question": question})
  
  return {"messages": generation}


def generation_no_history_stream(state: graph_state) -> Dict[str, Any]:
  """
  It is neccessary to explicitely add messages and write back to the
  state for streaming to work properly with MLflow. See the StreamState
  class documentation for more details.
  """
  question = choose_prompt_question(state)

  chain = prompt_no_history | model | RunnableLambda(format_generation_assistant)

  generation = chain.invoke({"context": state["context"],
                             "question": question})
  
  updated_messages = state["messages"] + generation

  return {"messages": updated_messages}


def generation_with_history(state: graph_state) -> Dict[str, Any]:
  """
  Ask the model to rephrase the user's question to incorporate message 
  history. This will result in a new question that will contain additional
  context helpful for retrieving documents from the vector index. 
  This rephrased question will be passed to the model to generate
  the final answer returned to the user.
  """  
  chain = prompt_with_history | model | RunnableLambda(format_generation_user)

  generation = chain.invoke({"messages": state['messages']})
  
  return {"generated_question": generation}




