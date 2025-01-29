"""Utility functions for loading images from Unity Catalog."""
from pathlib import Path
import tempfile
from typing import Optional, Union
from PIL import Image
import io

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import TableInfo
from databricks.sdk.service.files import DownloadResponse

def load_image_from_volume(
    catalog: str,
    schema: str,
    volume: str,
    image_path: str,
    workspace_client: Optional[WorkspaceClient] = None,
) -> Image.Image:
    """
    Load an image from a Unity Catalog Volume.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        volume: Volume name
        image_path: Path to image within the volume
        workspace_client: Optional WorkspaceClient instance
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image doesn't exist in volume
        ValueError: If image can't be opened
    """
    # Initialize workspace client if not provided
    if workspace_client is None:
        workspace_client = WorkspaceClient()
    
    # Construct the volume path
    volume_path = f"/Volumes/{catalog}/{schema}/{volume}/{image_path}"
    
    try:
        # Read file from volume
        with workspace_client.volumes.read(volume_path) as file:
            image_bytes = file.read()
            
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return image
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at path: {volume_path}")
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

def load_image_from_uc(
    catalog: str,
    schema: str,
    table: str,
    image_column: str,
    condition: Optional[str] = None,
    workspace_client: Optional[WorkspaceClient] = None,
) -> Image.Image:
    """
    Load an image from Unity Catalog.
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name
        image_column: Column containing the image data
        condition: Optional WHERE clause for filtering
        workspace_client: Optional WorkspaceClient instance
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If no image is found or multiple images are found
    """
    # Initialize workspace client if not provided
    if workspace_client is None:
        workspace_client = WorkspaceClient()
    
    # Construct the SQL query
    base_query = f"SELECT {image_column} FROM {catalog}.{schema}.{table}"
    if condition:
        query = f"{base_query} WHERE {condition}"
    else:
        query = f"{base_query} LIMIT 1"
    
    # Execute query
    result = workspace_client.sql.execute_and_fetch(query)
    
    # Check if we got exactly one result
    if not result:
        raise ValueError("No image found")
    if len(result) > 1:
        raise ValueError("Multiple images found, please provide a condition to select one image")
    
    # Get image bytes from first row
    image_bytes = result[0][image_column]
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    return image 

def load_image(
    volume_path: str,
    workspace_client: Optional[WorkspaceClient] = None,
) -> Image.Image:
    """
    Load an image from a Unity Catalog Volume using full path.
    
    Args:
        volume_path: Full path to image (e.g., '/Volumes/catalog/schema/volume/path/to/image.jpg')
        workspace_client: Optional WorkspaceClient instance
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image doesn't exist in volume
        ValueError: If image can't be opened
    """
    # Initialize workspace client if not provided
    if workspace_client is None:
        workspace_client = WorkspaceClient()
    
    try:
        # Read file from volume
        with workspace_client.volumes.read(volume_path) as file:
            image_bytes = file.read()
            
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return image
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at path: {volume_path}")
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

def response_to_image(response: DownloadResponse) -> Image.Image:
    """
    Convert a Databricks DownloadResponse to a PIL Image.
    
    Args:
        response: DownloadResponse from workspace_client.volumes.read()
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If image can't be opened
    """
    try:
        # Read the contents from the streaming response
        image_bytes = response.contents.read()
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return image
        
    except Exception as e:
        raise ValueError(f"Error converting response to image: {str(e)}") 