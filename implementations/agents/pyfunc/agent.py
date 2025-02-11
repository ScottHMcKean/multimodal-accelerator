import os
import mlflow
from mlflow.pyfunc import ChatModel

import mlflow
from maud.agent.config import parse_config

from maud.agent.retrievers import get_vector_retriever
from databricks_langchain import ChatDatabricks
from maud.agent.config import MaudConfig
from maud.agent.states import get_state
from maud.agent.nodes import (
    make_query_vector_database_node,
    make_context_generation_node,
)
from maud.agent.utils import format_chat_response_for_mlflow
from langgraph.graph import StateGraph, START, END


class GraphChatModel(ChatModel):
    """
    A mlflow ChatModel that invokes or streams a LangGraph
    workflow.

    The LangGraph output dictionary is parsed into mlflow chat model outputs
    that vary depending on whether the graph is invoked or streamed.

    MLflow autologging for Langchain must be explicitly turned on. It is not on
    by default as is the case when logging a langchain model flavor.
    """

    def __init__(self, config: MaudConfig = None):
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            self.mlflow_config = mlflow.models.ModelConfig(
                development_config=config_path
            )
            self.maud_config = parse_config(self.mlflow_config)
        else:
            self.maud_config = config

        self.make_graph()
        mlflow.langchain.autolog()

    def make_graph(self):
        retriever = get_vector_retriever(self.maud_config)
        model = ChatDatabricks(endpoint=self.maud_config.model.endpoint_name)

        state = get_state(self.maud_config)
        retriever_node = make_query_vector_database_node(retriever, self.maud_config)
        context_generation_node = make_context_generation_node(model, self.maud_config)

        # Graph
        workflow = StateGraph(state)
        workflow.add_node("retrieve", retriever_node)
        workflow.add_node("generate_w_context", context_generation_node)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate_w_context")
        workflow.add_edge("generate_w_context", END)
        self.app = workflow.compile()

    def predict_stream(self, context, messages, params=None):
        """
        NOTE: This method is not supported by Databricks model serving yet.
        Stream the application on the input messages. Selectively choose the output
        from graph events (node executions) to return to the user. This is necessary
        when the model is executed using the 'stream' method rather than the 'invoke'
        method. Each event in the stream represent the output of a LangGraph node.
        The event is a dictionary where the key is the node's name and the value is the
        nodes output dictionary, which conforms to the graph's state dictionary.

        The output from nodes that invoke the model are return to the user. The
        last event returned from the stream will also include the message history
        as a custom output within the mlflow chat message type.
        """
        if not isinstance(messages, dict):
            messages = {"messages": [message.to_dict() for message in messages]}

        # We only want to stream message events
        for event in self.app.stream(messages):
            # pull the event values out of the dictionary
            payload = next(iter(event.values()))
            
            # standard message response
            if 'messages' in payload:
                message_history = payload["messages"]
                answer = message_history[-1]["content"]
                yield format_chat_response_for_mlflow(
                    answer,
                    history=message_history,
                    stream=True,
                )

            # custome retriever response
            if 'documents' in payload:
                answer = payload["context"]
                yield format_chat_response_for_mlflow(
                    answer,
                    documents=payload["documents"],
                    stream=True,
                )

    def predict(self, context, messages, params=None):
        """
        Apply the invoke(batch) method to the LangGraph application. The entire graph will
        execute and the completed state dictionary will be returned by the model.

        Receives a list of messages in mlflow ChatMessage format. Convert the messages to a list
        of dictionaries before passing them to the application.

        Elements from the dictionary are retrieved and reformatted into the expected MLflow
        chat model type and are then returned.
        """
        if not isinstance(messages, dict):
            messages = {"messages": [message.to_dict() for message in messages]}
            
        generation = self.app.invoke(messages)
        message_history = generation["messages"]
        documents = generation.get("documents")
        answer = message_history[-1]["content"]
        return format_chat_response_for_mlflow(
            answer, history=message_history, documents=documents
        )


mlflow.models.set_model(GraphChatModel())
