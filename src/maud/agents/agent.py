import os
import mlflow
from operator import itemgetter
from mlflow.models import ModelConfig

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from databricks_langchain.vectorstores import DatabricksVectorSearch
from databricks_langchain.chat_models import ChatDatabricks

config = ModelConfig(development_config='../configs/agent_config.yaml')

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
    
vs_retriever = vector_search.as_retriever(
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

# Chain
prompt = PromptTemplate(
    template=config.get("system_prompt"), 
    input_variables=["context", "input"]
    )

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vs_retriever, combine_docs_chain)

mlflow.models.set_model(rag_chain)