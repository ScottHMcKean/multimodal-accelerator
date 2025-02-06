from typing import TypedDict, Annotated, List, Union
from operator import add
from maud.agent.config import MaudConfig

class StreamState(TypedDict):
  """
  Use this state when streaming is required. It is currently
  necessary to use the syncronous stream() method for MLflow 
  compatibility. Stream() only returns a node's output, not
  the updated state after the node's completion. Therefore,
  it is necessary to return the full updated message history
  from the node, rather than an update.
  """
  messages: List[dict[str,str]]
  generated_question: List[dict[str,str]]
  context: List[str] = [""]


class GraphState(TypedDict):
  messages: Annotated[List[dict[str,str]], add]
  generated_question: List[dict[str,str]]
  context: List[str] = [""]


def get_state(config: MaudConfig) -> Union[StreamState, GraphState]:
  """
  Load the proper state class depending on whether streaming
  inference is enabled.
  """
  return StreamState if config.agent.streaming else GraphState