import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing non-standard characters and spaces.

    Args:
        filename (str): The original filename to sanitize

    Returns:
        str: A sanitized filename with only alphanumeric characters, underscores, and dots
    """
    # Convert to Path object
    path = Path(filename)

    # Get stem (name without extension) and suffix (extension)
    stem = path.stem
    suffix = path.suffix

    # Replace spaces and special characters with underscores
    # Keep only alphanumeric characters, underscores, and dots
    sanitized = re.sub(r"[^\w\s-]", "", stem)

    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Add back the extension
    return f"{sanitized}{suffix}"


# Example usage:
# sanitize_filename("My Document (2023).pdf")  # Returns: "My_Document_2023.pdf"
# sanitize_filename("File@Name#123.txt")       # Returns: "FileName123.txt"
# sanitize_filename("  Multiple   Spaces  .docx")  # Returns: "Multiple_Spaces.docx"
