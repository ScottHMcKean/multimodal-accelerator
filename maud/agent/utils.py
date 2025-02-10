from functools import partial
from typing import List, Dict, Iterator, Union
from mlflow.types.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
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
) -> Dict:
    """
    Reformat the applications responses to conform to the ChatCompletionResponse
    required by Databricks Mosaic AI Agent Framework
    """
    return asdict(
        ChatCompletionResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=answer))],
            custom_outputs={
                "message_history": history,
                "documents": documents,
            },
        )
    )


def convert_to_chat_request(
    messages: List[Dict[str, setattr]]
) -> ChatCompletionRequest:
    """
    Convert a messages to ChatCompletionRequest format. Input messages should conform
    to the below format.

    [
      {'role': 'user', 'content': 'What is Apache Spark'},
      {'role': 'assistant', 'content': 'Apache Spark is a distributed data processing engine.'}
    ]

    Relavent docs:
      https://mlflow.org/docs/latest/llms/chat-model-intro/index.html#tutorial-getting-started-with-chatmodel
      https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatCompletionRequest
      https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatMessage
      https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatParams
      https://mlflow.org/docs/latest/llms/chat-model-intro/index.html#building-a-chatmodel-that-accepts-inference-parameters
    """
    chat_messages = []
    for message in messages:
        chat_messages.append(
            ChatMessage(content=message["content"], role=message["role"])
        )

    chat_request = ChatCompletionRequest(messages=chat_messages)
    return chat_request.messages


def print_generation_and_history(chat_responses: List, i: int, streaming: bool = False):
    """
    Accepts a ChatCompletionResponse and and print its contents. This function
    is used to analyzing model outputs in a notebook before logging the
    model to MLflow.
    """
    if streaming:
        # Streaming responses can contain two events, where the first event
        # contains the question rewrite response, and the second event
        # contains the final generation and message history.
        multiple_events = True if len(chat_responses[i]) > 1 else False
        print(f"{chat_responses[i][0].choices[0].delta}\n")
        if multiple_events:
            print(f"{chat_responses[i][1].choices[0].delta}\n")
            print(f"{chat_responses[i][1].custom_outputs}\n")
        else:
            print(f"{chat_responses[i][0].custom_outputs}\n")
    else:
        print(f"{chat_responses[i].choices[0].message}\n")
        print(chat_responses[i].custom_outputs)


def format_chat_response_for_mlflow(answer, history=None, documents=None, stream=False):
    """
    Reformat the LangGraph dictionary output into mlflow chat model types.
    Streaming output requires the ChatCompletionChunk type; batch (invoke)
    output requires the ChatCompletionResponse type.

    The models answer to the users question is returned. The messages history,
    if it exists, is returned as a custom output within the chat message type.
    """
    if stream:
        chat_completion_response = ChatCompletionChunk(
            choices=[
                ChatChunkChoice(delta=ChatChoiceDelta(role="assistant", content=answer))
            ]
        )

    else:
        chat_completion_response = ChatCompletionResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=answer))]
        )

    if history:
        chat_completion_response.custom_outputs = {
            "message_history": history,
            "documents": documents,
        }

    return chat_completion_response
