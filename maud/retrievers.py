import mlflow
from databricks_langchain import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from config_utils import load_config

config = load_config("retriever")
vector_search_schema = config.get("schema")
embedding_model = DatabricksEmbeddings(endpoint=config.get("embedding_model"))

# Docs: https://python.langchain.com/docs/integrations/providers/databricks/#vector-search
vector_search_retriever = DatabricksVectorSearch(
    endpoint = config.get("vector_search_endpoint_name"),
    index_name = config.get("vector_search_index"),
    text_column = vector_search_schema.get("chunk_text"),
    embedding=embedding_model,
    columns =[
      vector_search_schema.get("primary_key"),
      vector_search_schema.get("chunk_text"),
      vector_search_schema.get("document_uri")
    ]).as_retriever(search_kwargs=config.get("parameters"))


def format_documents(docs):
    chunk_template = config.get("chunk_template")
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata[vector_search_schema.get("document_uri")],
        )
        for d in docs
    ]
    return "".join(chunk_contents)

def index_exists(client, vs_endpoint, index_name):
    try:
        index_info = client.get_index(vs_endpoint, index_name)
        return True
    except Exception as e:
        if 'IndexNotFoundException' in str(e):
            return False
        else:
            raise e