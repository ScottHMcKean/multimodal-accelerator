import logging
import io
import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import os
from databricks.vector_search.client import VectorSearchClient
from maud.configs import Config
import json
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
workspace_client = WorkspaceClient()

# Load configuration
config = Config("app_maud_config.yaml")

def query_vector_search(message, history):
    """
    Query the LLM with the given message and chat history.
    """
    logger.info(f"Message: {message}")
    if isinstance(message, dict):
        message = message.get("text", "")

    if not isinstance(message, str) or not message.strip():
        return "ERROR: The question should not be empty"

    messages = []
    
    # Add history messages
    # for msg in history:
    #     messages.append(ChatMessage(
    #         role=ChatMessageRole.USER if msg["role"] == "user" else ChatMessageRole.ASSISTANT,
    #         content=msg["content"]
    #     ))

    messages.append(ChatMessage(role=ChatMessageRole.USER, content=message))

    logger.info(f"Sending request to model endpoint: {config.serving_endpoint}")

    vsc = VectorSearchClient(
        workspace_url=workspace_url,
        service_principal_client_id=sp_client_id,
        service_principal_client_secret=sp_client_secret
    )

    index = vsc.get_index(endpoint_name="one-env-shared-endpoint-3", index_name="shm.multimodal.multimodal_index")

    # Response to parse (already a dict)
    sim_search_results = index.similarity_search(
        num_results=3, 
        columns=["text"], 
        query_text="example_query"
        )
    
    logger.info("Received response from vector search")
    logger.info(response)
    
    gradio_messages = []
    
    # Parse the response JSON
    response_str = json.loads(sim_search_results)

    gradio_messages.append({
        'text':response_dict['response'],
        })
    logger.info(gradio_messages)
    return gradio_messages

demo = gr.ChatInterface(
    fn=query_vector_search, 
    type="messages",
    multimodal=True,
    )

if __name__ == "__main__":
    demo.launch()