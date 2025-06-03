import logging
import os
import io
from pathlib import Path
import streamlit as st
from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient
from PIL import Image

# Get workspace client
workspace_client = WorkspaceClient()
workspace_url = workspace_client.config.host
token = workspace_client.config.token

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

# Use info (prep for passthrough auth)
def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )
user_info = get_user_info()

# Brand Colors
PRIMARY = "#ACACAC"   # Deep blue
ACCENT = "#E22E2F"    # Accent blue
BG = "#F5F6FA"        # Light background

st.set_page_config(
    page_title="Multimodal Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS
st.markdown(f"""
    <style>
        .main {{
            background-color: {BG};
        }}
        .block-container {{
            padding-top: 3.5rem;
        }}
        .maud-header {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
            background: white;
            border-radius: 8px;
            padding: 1.2rem 2rem 1.2rem 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 8px rgba(0,0,0,0.03);
        }}
        .maud-title {{
            color: {PRIMARY};
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
            margin-top: 0.1rem;
        }}
        .maud-subtitle {{
            color: {ACCENT};
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 0;
        }}
        .stChatMessage {{
            background: #fff;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            padding: 1rem;
            border: 1px solid #e0e0e0;
        }}
        .stSidebar > div:first-child {{
            background: {PRIMARY};
            color: white;
        }}
        .stSidebar .stHeader {{
            color: white;
        }}
    </style>
""", unsafe_allow_html=True)

# Welcome
st.markdown(
    """
    Welcome to the MAUD Chatbot!
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "referenced_documents" not in st.session_state:
    st.session_state.referenced_documents = set()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know about today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Query the Databricks serving endpoint
        result = get_deploy_client('databricks').predict(
            endpoint=os.getenv("SERVING_ENDPOINT"),
            inputs={'messages': st.session_state.messages, "max_tokens": 10000},
        )
        assistant_response = result['messages'][-1]['content']
        st.markdown(assistant_response)

    # Handle custom_outputs for images and pages
    custom_outputs = result.get("custom_outputs", {})
    st.session_state.referenced_documents = custom_outputs.get('documents',[])
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

with st.sidebar:
    st.header("Retrieved References")
    
    if st.session_state.referenced_documents:
        for doc in st.session_state.referenced_documents:
            img_path = doc['metadata']['image_path']
            page = int(doc['metadata']['pages'][0])
            filename = doc['metadata']['filename']
            
            if img_path == '':
                continue
            
            img = load_image(img_path)
            
            if img:
                st.markdown(
                    f"**{filename.lower()}** &mdash; Page {page}"
                )
                st.image(img, use_container_width=True)