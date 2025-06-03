# Config
import mlflow
from maud.agent.config import parse_config
mlflow_config = mlflow.models.ModelConfig(development_config="./config.yaml")
maud_config = parse_config(mlflow_config)

# API Interfaces
from maud.agent.retrievers import get_vector_retriever
from databricks_langchain import ChatDatabricks
retriever = get_vector_retriever(maud_config)
model = ChatDatabricks(endpoint=maud_config.model.endpoint_name)

# Nodes
from maud.agent.states import get_state
from maud.agent.nodes import (
    make_query_vector_database_node,
    make_context_generation_node,
)

state = get_state(maud_config)
retriever_node = make_query_vector_database_node(retriever, maud_config)
context_generation_node = make_context_generation_node(model, maud_config)

# Graph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from maud.agent.utils import graph_state_to_chat_type

workflow = StateGraph(state)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("generate_w_context", context_generation_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate_w_context")
workflow.add_edge("generate_w_context", END)

graph = workflow.compile()

# MLFLow Wrapper
from typing import Any, Generator, Optional, Sequence, Union
from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import uuid

class MAUDAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
    
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }

        messages = []
        custom_outputs = {}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                msgs = node_data.get("messages", [])
                for msg in msgs:
                    msg.update({'id':str(uuid.uuid4())})
                messages.extend([ChatAgentMessage(**msg) for msg in msgs])
                if node_data.get("documents"):
                    custom_outputs["documents"] = [
                        doc.model_dump() for doc 
                        in node_data.get("documents")
                        ]
        
        return ChatAgentResponse(
            messages=messages, 
            custom_outputs=custom_outputs
            )

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        
        custom_outputs = {}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values(): 
                msgs = node_data.get("messages", [])
                [msg.update({'id':str(uuid.uuid4())}) for msg in msgs]
                if node_data.get("documents"):
                    custom_outputs["documents"] = [
                        doc.model_dump() for doc 
                        in node_data.get("documents")
                        ]
                yield from (
                    ChatAgentChunk(
                        delta=msg, 
                        custom_outputs=custom_outputs)
                        for msg in msgs
                )

mlflow.langchain.autolog()
AGENT = MAUDAgent(graph)
mlflow.models.set_model(AGENT)