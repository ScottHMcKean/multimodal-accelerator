"""Test configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path

# Test configuration
# pytest_plugins = ["pytest_asyncio"]  # Disabled - not using async tests currently


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_pdf_path(temp_directory):
    """Create a sample PDF file path for testing."""
    return temp_directory / "sample.pdf"


@pytest.fixture
def sample_docx_path(temp_directory):
    """Create a sample DOCX file path for testing."""
    return temp_directory / "sample.docx"


@pytest.fixture 
def test_filenames():
    """Provide test filenames for sanitization testing."""
    return [
        "normal_file.pdf",
        "file with spaces.docx", 
        "file<>with|bad:chars.xlsx",
        "My%20Document.pdf",
        "document_(final).txt"
    ]


@pytest.fixture
def processing_config():
    """Create a test processing configuration."""
    from src.core import ProcessingConfig
    return ProcessingConfig("test", "default", "input", "output")


@pytest.fixture
def sample_processing_results():
    """Provide sample processing results for testing."""
    return [
        {"status": "success", "file": "document1.pdf", "pages": 10},
        {"status": "success", "file": "document2.docx", "pages": 5},
        {"status": "error", "file": "document3.pdf", "error": "Parsing failed"}
    ]


@pytest.fixture
def create_test_files():
    """Factory fixture to create test files."""
    def create_files(directory: Path, filenames: list):
        """Create test files in the given directory."""
        files = []
        for filename in filenames:
            file_path = directory / filename
            file_path.write_text(f"Test content for {filename}")
            files.append(file_path)
        return files
    return create_files