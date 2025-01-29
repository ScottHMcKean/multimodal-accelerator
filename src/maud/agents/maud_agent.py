import mlflow
from mlflow.models import ModelConfig

from databricks_langchain.vectorstores import DatabricksVectorSearch
from databricks_langchain.chat_models import ChatDatabricks

from typing import List, TypedDict, Annotated, Dict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph

config = ModelConfig(development_config='../configs/agent_config.yaml')

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: dict

# Foundation Model
llm_config = config.get("llm")

llm = ChatDatabricks(
    endpoint=llm_config.get("endpoint_name"), 
    max_tokens=llm_config.get("max_tokens"), 
    temperature=llm_config.get("temperature")
    )

# Vector Search
vs_config = config.get("vector_search")

vector_search = DatabricksVectorSearch(
    endpoint=vs_config.get("endpoint_name"),
    index_name=vs_config.get("index_name"),
    columns=[
        vs_config.get('primary_key'), 
        vs_config.get('text_column'), 
        vs_config.get('doc_uri')
    ] + vs_config.get("other_columns", [])
)
    
retriever = vector_search.as_retriever(
    search_type=vs_config.get("search_type"), 
    search_kwargs={
        'k': vs_config.get('num_results'), 
        "score_threshold": vs_config.get('score_threshold'),
        'query_type': vs_config.get('query_type')
        }
)

mlflow.models.set_retriever_schema(
    primary_key= vs_config.get('primary_key'),
    text_column= vs_config.get('text_column'),
    doc_uri= vs_config.get('doc_uri'),
    name="vs_index",
)

system_prompt = config.get("system_prompt")

def retrieve(state: State):
    last_message = state["messages"][-1]
    docs = retriever.invoke(last_message.content)

    merged_doc_dict = [doc.to_json() for doc in docs]

    context = "\n\n".join(
        f"ID: {int(doc.metadata['id'])}\nContent: {doc.page_content}\nSource: {doc.metadata['img_path']}"
        for doc in docs
    )
    
    return {
        "messages": [SystemMessage(content=system_prompt+context)],
        "documents": merged_doc_dict
    }


def generate(state: State):
    messages = state['messages']
    system_message = [m for m in messages if m.type == 'system']
    messages_for_llm = system_message + [m for m in messages if m.type=="human"]
    response = llm.invoke(messages_for_llm)
    return {"messages": [response]}

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import json

def parse_messages_docs(state) -> str:
    last_message = state["messages"][-1]
    parser = StrOutputParser()
    parsed_output = parser.parse(last_message.content)

    documents: dict= state.get("documents",{})
    
    # Create a JSON object
    output = {
        "response": parsed_output, 
        "documents": documents
        }
    
    # Convert the dictionary to a JSON string
    return json.dumps(output, indent=2)

graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")

graph = graph.compile()
chain = graph | RunnableLambda(parse_messages_docs)

mlflow.models.set_model(chain)