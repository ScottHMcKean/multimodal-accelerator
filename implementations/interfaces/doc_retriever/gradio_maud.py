import logging
import io
import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from maud.config import Config
import json
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
workspace_client = WorkspaceClient()

# Load configuration
config = Config("app_maud_config.yaml")


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


def query_llm(message, history):
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
    response = workspace_client.serving_endpoints.query(
        name=config.serving_endpoint, messages=messages
    )
    logger.info("Received response from model endpoint")
    logger.info(response)
    gradio_messages = []

    # Parse the response JSON
    response_dict = json.loads(response.choices[0].message.content)

    gradio_messages.append(
        {
            "text": response_dict["response"],
        }
    )

    documents = response_dict["documents"]
    for doc in documents:
        metadata = doc.get("kwargs", {}).get("metadata", {})
        img_path = metadata.get("img_path", "")
        if img_path:
            print(f"loading image from {img_path}")
            img = load_image(img_path)
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
