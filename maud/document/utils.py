from pathlib import Path
import re

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing non-standard characters and spaces.
    
    Args:
        filename (str): The original filename to sanitize
        
    Returns:
        str: A sanitized filename with only alphanumeric characters, underscores, and dots
    """
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    
    stem = stem.replace('%20', '_')    
    sanitized = re.sub(r'[^\w\s-]', '', stem)
    sanitized = sanitized.replace(' ', '_')
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_').lower()

    return f"{sanitized}{suffix}"