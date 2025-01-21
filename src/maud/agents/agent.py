import os
import mlflow
from mlflow.models import ModelConfig

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import TransformChain
from langchain.chains import SequentialChain

from databricks_langchain.vectorstores import DatabricksVectorSearch
from databricks_langchain.chat_models import ChatDatabricks

config = ModelConfig(development_config='src/maud/config/default/agent_config.yaml')

# Foundation Model
llm_config = config.get("llm")

chat_model = ChatDatabricks(
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
    template=config.get("retrieval_prompt"), 
    input_variables=["context", "question"]
    )

retrieval_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vs_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

mlflow.models.set_model(retrieval_chain)