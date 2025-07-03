"""Shared test fixtures and configuration for the MAUD test suite."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock

import pytest
import pandas as pd
from PIL import Image
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.document_converter import PdfFormatOption

# Mock imports to prevent heavy dependencies during testing
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_output"
        output_path.mkdir(exist_ok=True)
        yield output_path


@pytest.fixture
def sample_pdf_path(test_data_dir) -> Path:
    """Return path to a sample PDF file."""
    return (
        test_data_dir / "maintenance_procedure_template.docx"
    )  # Smaller file for speed


@pytest.fixture
def sample_docx_path(test_data_dir) -> Path:
    """Return path to a sample DOCX file."""
    return test_data_dir / "maintenance_procedure_template.docx"


@pytest.fixture
def sample_pptx_path(test_data_dir) -> Path:
    """Return path to a sample PPTX file."""
    return test_data_dir / "functional_flight_checks.pptx"


@pytest.fixture
def sample_xlsx_path(test_data_dir) -> Path:
    """Return path to a sample XLSX file."""
    return test_data_dir / "equipment_maintenance_schedule.xlsx"


@pytest.fixture
def small_test_image() -> Image.Image:
    """Create a small test image for fast testing."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a test description."
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_docling_document():
    """Create a mock DoclingDocument for testing."""
    mock_doc = Mock(spec=DoclingDocument)
    mock_doc.pages = {0: Mock()}
    mock_doc.pages[0].page_no = 0
    mock_doc.pages[0].image = Mock()
    mock_doc.pages[0].image.pil_image = Image.new("RGB", (100, 100), color="blue")
    mock_doc.pictures = []
    mock_doc.tables = []
    mock_doc.model_dump.return_value = {"test": "data"}
    return mock_doc


@pytest.fixture
def sample_chunk_data() -> list[dict]:
    """Sample chunk data for testing."""
    return [
        {
            "filename": "test.pdf",
            "input_hash": "abc123",
            "pages": [0],
            "doc_refs": ["ref1"],
            "has_table": False,
            "has_picture": False,
            "tables": [],
            "pictures": [],
            "headings": ["Introduction"],
            "captions": [],
            "chunk_type": "text",
            "image_path": "",
            "text": "This is sample text content.",
            "enriched_text": "This is enriched sample text content.",
        },
        {
            "filename": "test.pdf",
            "input_hash": "abc123",
            "pages": [0],
            "doc_refs": ["ref2"],
            "has_table": True,
            "has_picture": False,
            "tables": ["table1"],
            "pictures": [],
            "headings": [],
            "captions": ["Table 1: Sample data"],
            "chunk_type": "table",
            "image_path": "/path/to/table.webp",
            "text": "Column1 | Column2\nValue1 | Value2",
            "enriched_text": "Table showing Column1 and Column2 data.",
        },
    ]


@pytest.fixture
def mock_databricks_agent():
    """Mock Databricks agent for testing."""
    mock_agent = Mock()
    mock_agent.invoke.return_value = {"output": "Test agent response"}
    return mock_agent


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    mock_mlflow = Mock()
    mock_mlflow.start_run.return_value = Mock()
    mock_mlflow.log_param = Mock()
    mock_mlflow.log_metric = Mock()
    mock_mlflow.log_artifact = Mock()
    return mock_mlflow


@pytest.fixture
def fast_pipeline_options():
    """Pipeline options optimized for fast testing."""
    from src.document.converters import MAUDPipelineOptions

    return MAUDPipelineOptions(
        generate_page_images=False,  # Skip image generation for speed
        generate_picture_images=False,
        generate_table_images=False,
        do_page_description=False,
        do_picture_description=False,
        do_picture_classification=False,
        images_scale=1.0,  # Minimal scale
    )


@pytest.fixture
def comprehensive_pipeline_options():
    """Full pipeline options for comprehensive testing."""
    from src.document.converters import MAUDPipelineOptions

    return MAUDPipelineOptions(
        generate_page_images=True,
        generate_picture_images=True,
        generate_table_images=True,
        do_page_description=False,  # Skip LLM calls for speed
        do_picture_description=False,
        do_picture_classification=False,
        images_scale=1.0,
    )


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Automatically mock external services for all tests."""
    # Mock OpenAI calls
    mock_openai = Mock()
    mock_openai.chat.completions.create.return_value = Mock()
    mock_openai.chat.completions.create.return_value.choices = [Mock()]
    mock_openai.chat.completions.create.return_value.choices[0].message.content = (
        "Mocked response"
    )

    # Mock Databricks services
    mock_databricks = Mock()

    # Apply mocks
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def performance_test_data():
    """Data for performance testing."""
    return {
        "small_text": "Short text for testing" * 10,
        "medium_text": "Medium length text for testing performance" * 100,
        "large_text": "Large text content for performance testing" * 1000,
    }


class TestHelpers:
    """Test helper methods."""

    @staticmethod
    def create_test_file(path: Path, content: str = "test content"):
        """Create a test file with content."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    @staticmethod
    def assert_file_exists(path: Path):
        """Assert that a file exists."""
        assert path.exists(), f"File {path} does not exist"

    @staticmethod
    def assert_directory_structure(base_path: Path, expected_dirs: list[str]):
        """Assert expected directory structure exists."""
        for dir_name in expected_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Directory {dir_name} not found in {base_path}"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers
