def format_documents(config, docs):
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