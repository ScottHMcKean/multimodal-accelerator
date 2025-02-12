import logging
import io
from pathlib import Path
import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from maud.interface.config import load_config
import json
from PIL import Image
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

workspace_client = WorkspaceClient()
workspace_url = workspace_client.config.host
token = workspace_client.config.token

app_config = load_config(str(Path(__file__).parent / "app_config.yaml"))


def load_image(volume_path: str) -> Image.Image:
    """
    Load an image from a Unity Catalog Volume using full path.

    Args:
        volume_path: Full path to image (e.g., '/Volumes/catalog/schema/volume/path/to/image.jpg')

    Returns:
        PIL.Image.Image: The loaded image if successful
        None: If there was an error loading the image

    Raises:
        FileNotFoundError: If image doesn't exist in volume
        ValueError: If image format is invalid or corrupted
        IOError: If there are permission issues or other I/O errors
    """
    try:
        response = workspace_client.files.download(volume_path)
        image_bytes = response.contents.read()
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        logging.warning(f"Error loading image: {str(e)}")
        return None


def clean_text(text: str) -> str:
    """Clean text by removing newlines and special characters."""
    import re

    # Remove newlines
    text = text.replace("\n", " ")
    # Remove special characters but keep spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Remove extra spaces
    text = " ".join(text.split())
    return text


def query_llm(message, history):
    """
    Query the LLM with the given message and chat history.
    """
    logger.info(f"Message: {message}")
    logger.info(f"History: {history}")

    if isinstance(message, dict):
        message_text = message.get("text", "")
    elif isinstance(message, str):
        message_text = message
    else:
        raise NotImplementedError(f"Unsupported message type: {type(message)}")

    endpoint_url = (
        f"{workspace_url}/serving-endpoints/{app_config.serving_endpoint}/invocations"
    )

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "messages": [
            {"role": "user", "content": message_text},
        ]
    }

    response = requests.post(endpoint_url, headers=headers, json=payload)

    # Check response status and get result
    if response.status_code == 200:
        result = response.json()
        logger.info(result)
    else:
        logger.error(f"Error: {response.status_code}")
        logger.error(response.text)

    gradio_messages = []
    gradio_messages.append(
        {
            "text": result["choices"][0]["message"]["content"],
        }
    )

    if result.get("custom_outputs").get("documents"):
        documents = result.get("custom_outputs").get("documents")
        for doc in documents:
            metadata = doc.get("metadata", {})
            content = doc.get("page_content", "")

            formatted_text = f"""
            Source: {metadata.get('filename', 'Unknown')}
            Type: {metadata.get('type', 'Unknown')}
            ID: {metadata.get('id', 'Unknown')}
            Content: {clean_text(content[0:250])}
            """
            gradio_messages.append(
                {"text": formatted_text, "is_text_box": True, "lines": 5}
            )

            if metadata.get("img_path"):
                print(f"loading image from {metadata.get('img_path')}")
                img = load_image(metadata.get("img_path"))
                if img:
                    gradio_messages.append(gr.Image(img))

    logger.info(gradio_messages)
    return gradio_messages


demo = gr.ChatInterface(
    fn=query_llm,
    type="messages",
    multimodal=True,
)

if __name__ == "__main__":
    demo.launch()
