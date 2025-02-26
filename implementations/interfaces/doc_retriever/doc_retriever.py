import logging
import io
from pathlib import Path
import requests


import streamlit as st
from databricks.sdk import WorkspaceClient
from PIL import Image

from maud.interface.config import load_config
from maud.interface.highlighting import highlight_stemmed_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

workspace_client = WorkspaceClient()
workspace_url = workspace_client.config.host
token = workspace_client.config.token

config = load_config(str(Path(__file__).parent / "config.yaml"))


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


def main():
    st.title(config.title)
    st.write(config.description)

    # Query input
    query = st.text_input(
        label="Enter your query:",
        placeholder=config.example,
    )

    endpoint_url = (
        f"{workspace_url}/serving-endpoints/{config.serving_endpoint}/invocations"
    )

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "messages": [
            {"role": "user", "content": query},
        ]
    }

    if query:
        with st.spinner("Processing query..."):
            try:
                response = requests.post(endpoint_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes
                result = response.json()

                st.write("### Assistant")
                ai_response = result["choices"][0]["message"]["content"]
                query_terms = query.lower().split()
                st.markdown(ai_response, unsafe_allow_html=True)

                # Display retrieved documents
                if "documents" in result["custom_outputs"]:
                    st.write("### Retrieved Documents")

                    for doc in result["custom_outputs"].get("documents", []):
                        metadata = doc.get("metadata", {})
                        content = doc.get("page_content", "")
                        with st.expander(f"Document {metadata.get('id', '')}"):
                            # Display metadata
                            st.markdown("**Metadata:**")
                            st.json(doc.get("metadata", {}))

                            # Display content
                            highlighted_text = highlight_stemmed_text(
                                clean_text(content), query_terms
                            )
                            st.markdown(highlighted_text, unsafe_allow_html=True)

                            # Display images
                            if metadata.get("img_path"):
                                img = load_image(metadata.get("img_path"))
                                if img:
                                    st.image(img, caption="Retrieved Image")

                            # Add Document Link
                            if "doc_id" in doc:
                                st.markdown(
                                    f"[View Original Document](document/{metadata.get('url','')})"
                                )

            except requests.exceptions.RequestException as e:
                st.error(f"Error making request: {str(e)}")


if __name__ == "__main__":
    main()
