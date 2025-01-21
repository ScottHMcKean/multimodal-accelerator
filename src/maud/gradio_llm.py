import os
from typing import Optional
import logging
import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
workspace_client = WorkspaceClient()
serving_endpoint = 'shm-gpt-4o-mini'

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
        logger.info(f"Sending request to model endpoint: {serving_endpoint}")
        response = workspace_client.serving_endpoints.query(
            name=serving_endpoint,
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
    title="Databricks LLM Chat",
    description="Chat with a Databricks-hosted LLM model",
    examples=[
        ["What is machine learning?"],
        ["Write a short poem about data science"],
        ["Explain what a neural network is"],
    ],
    type="messages"
)

if __name__ == "__main__":
    demo.launch()