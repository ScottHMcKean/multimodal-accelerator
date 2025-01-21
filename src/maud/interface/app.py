import gradio as gr
import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.vector_search.client import VectorSearchClient
import os
import re
from PIL import Image
import io
from maud.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = Config("app_config.yaml")

# Initialize clients
workspace_client = WorkspaceClient()
vs_client = VectorSearchClient()

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

def vector_search(query, vs_endpoint_name=None, index_name=None):
    """Perform vector search using Databricks Vector Search"""
    vs_endpoint_name = vs_endpoint_name or config.vector_search.endpoint_name
    index_name = index_name or config.vector_search.index_name
    
    try:
        response = vs_client.get_index(
            endpoint_name=vs_endpoint_name,
            index_name=index_name
        ).similarity_search(
            query_text=query,
            num_results=config.vector_search.num_results
        )
        return response.results
    except Exception as e:
        logger.error(f"Vector search error: {str(e)}")
        return []
    
def download_image_from_volume(image_path):
    """
    Download image from Databricks volume
    """
    try:
        with workspace_client.dbfs.read(f"/Volumes/{image_path}") as f:
            image_data = f.read()
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return None

def extract_image_urls(text):
    """
    Extract image URLs from text using regex
    """
    pattern = r'/Volumes/[^\s)"]+'
    return re.findall(pattern, text)

def query_llm(message, history):
    """
    Query the LLM with the given message and chat history.
    """
    if not message.strip():
        return "ERROR: The question should not be empty", []

    # Perform vector search using Databricks Vector Search
    search_results = vector_search(message)
    
    # Extract content and sources from search results
    context_parts = []
    image_urls = []
    
    for result in search_results:
        # Add content to context
        context_parts.append(result.content)
        
        # Extract image URLs from content
        urls = extract_image_urls(result.content)
        for url in urls:
            img = download_image_from_volume(url)
            if img:
                image_urls.append(img)

    context = "\n".join(context_parts)
    
    prompt = f"""Answer this question like a helpful assistant. Use the following context:
    {context}
    
    Question: {message}"""
    
    messages = [ChatMessage(role=ChatMessageRole.USER, content=prompt)]

    try:
        logger.info(f"Sending request to model endpoint: {os.getenv('SERVING_ENDPOINT')}")
        response = workspace_client.serving_endpoints.query(
            name=os.getenv('SERVING_ENDPOINT'),
            messages=messages,
            max_tokens=400
        )
        logger.info("Received response from model endpoint")
        return response.choices[0].message.content, image_urls
    except Exception as e:
        logger.error(f"Error querying model: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", []

# Create Gradio interface with sidebar for images
with gr.Blocks() as demo:
    gr.Markdown("# Databricks LLM Chatbot")
    gr.Markdown("Ask questions and get responses from a Databricks LLM model.")
    
    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.ChatInterface(
                fn=query_llm,
                examples=[
                    "What is machine learning?",
                    "What are Large Language Models?",
                    "What is Databricks?"
                ],
            )
        with gr.Column(scale=3):
            gallery = gr.Gallery(
                label="Retrieved Images",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                height="auto"
            )
            
    chatbot.submit(fn=None, inputs=chatbot.textbox, outputs=[chatbot.chatbot, gallery])

if __name__ == "__main__":
    demo.launch()