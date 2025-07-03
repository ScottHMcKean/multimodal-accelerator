from functools import partial
from typing import List, Dict, Iterator, Union
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.messages.utils import (
    convert_to_messages,
    convert_to_openai_messages,
)
from langgraph.graph import StateGraph
from dataclasses import asdict

from mlflow.types.llm import (
    ChatMessage,
    ChatCompletionResponse,
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
)


def format_generation(role: str, generation) -> Dict[str, str]:
    """
    Reformat Chat model response to a list of dictionaries. This function
    is called within the graph's nodes to ensure a consistent chat
    format is saved in the graph's state
    """
    return [{"role": role, "content": generation.content}]


format_generation_user = partial(format_generation, "user")
format_generation_assistant = partial(format_generation, "assistant")


def get_last_user_message(state: StateGraph) -> List[Dict[str, str]]:
    """
    Return the last user message from the state.
    Uses LangChain's convert_to_messages and convert_to_openai_messages
    functions to convert the state to a list of dictionaries with
    'role' and 'content' keys back into the state.
    """
    messages = convert_to_messages(state["messages"])
    last_msg = [[x for x in messages if x.type == "human"][-1]]
    return convert_to_openai_messages(last_msg)


def graph_state_to_chat_type(state: StateGraph):
    """
    Reformat the applications responses to conform to the ChatCompletionResponse
    required by Databricks Mosaic AI Agent Framework. This function can be applied
    to langgraph graphs called via 'invoke' and applied via RunnableLambda

    chain = compile_graph | RunnableLambda(graph_state_to_chat_type)
    """
    answer = state["messages"][-1]["content"]

    # Add history
    history = []
    if len(state["messages"]) > 1:
        history += state["messages"][:-1]

    if "context" in state:
        history += [{"role": "tool", "content": state["context"]}]

    documents = []
    if "documents" in state:
        documents = [x.model_dump() for x in state["documents"]]

    return create_flexible_chat_completion_response(answer, history, documents)

def create_flexible_chat_completion_response(
    answer: str,
    history: List[Dict[str, str]] = None,
    documents: List[Dict[str, str]] = None,
) -> ChatAgentResponse:
    """
    Reformat the applications responses to conform to the ChatCompletionResponse
    required by Databricks Mosaic AI Agent Framework
    """
    return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=answer)],
            custom_outputs={
                "message_history": history,
                "documents": documents,
            },
        )