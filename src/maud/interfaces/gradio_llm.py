import os
from typing import Optional
import logging
import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from maud.configs import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
workspace_client = WorkspaceClient()

# Load configuration
config = Config("app_config.yaml")

def query_llm(message, history):
    """
    Query the LLM with the given message and chat history.
    """
    if not message.strip():
        return "ERROR: The question should not be empty"

    # Convert history and new message into ChatMessage format
    messages = []
    
    # Add history messages
    for msg in history:
        messages.append(ChatMessage(
            role=ChatMessageRole.USER if msg["role"] == "user" else ChatMessageRole.ASSISTANT,
            content=msg["content"]
        ))
    
    # Add the new message
    messages.append(ChatMessage(role=ChatMessageRole.USER, content=message))

    try:
        logger.info(f"Sending request to model endpoint: {config.serving_endpoint}")
        response = workspace_client.serving_endpoints.query(
            name=config.serving_endpoint,
            messages=messages,
            max_tokens=400
        )
        logger.info("Received response from model endpoint")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying model: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=query_llm,
    title=config.interface.title,
    description=config.interface.description,
    examples=config.interface.examples,
    type="messages"
)

if __name__ == "__main__":
    demo.launch()