from pathlib import Path
import shutil
import logging

import pytest
import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

from maud.document.converters import MAUDConverter
from maud.document.converters import MAUDPipelineOptions, MAUDConverter, MAUDPipeline


@pytest.fixture(scope="session")
def output_dir():
    """Create output directory and clean it up after tests."""
    path = Path("tests/data/output")
    path.mkdir(exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for all tests."""
    logging.basicConfig(level=logging.INFO)
    yield


def test_pipeline_options_validation():
    """Test that pipeline options validation works correctly."""
    # Test valid options
    valid_options = MAUDPipelineOptions(
        do_page_description=True,
        generate_page_images=True,
    )
    assert valid_options.do_page_description is True
    assert valid_options.generate_page_images is True

    # Test invalid options
    with pytest.raises(
        ValueError,
        match="do_page_description requires generate_page_images to be enabled",
    ):
        MAUDPipelineOptions(
            do_page_description=True,
            generate_page_images=False,
        )


def test_converter_instantiation():
    converter = MAUDConverter(
        input_path=Path("tests/data/wind_turbine.pdf"),
        output_dir=Path("tests/data/output"),
    )
    assert isinstance(converter, MAUDConverter)
    assert isinstance(converter.input_path, Path)
    assert isinstance(converter.output_dir, Path)


@pytest.mark.slow
def test_convert_pdf():
    converter = MAUDConverter(
        input_path=Path("tests/data/wiring_bonding.pdf"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


def test_convert_docx():
    converter = MAUDConverter(
        input_path=Path("tests/data/maintenance_procedure_template.docx"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


def test_convert_pptx():
    converter = MAUDConverter(
        input_path=Path("tests/data/functional_flight_checks.pptx"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


def test_convert_xlsx():
    converter = MAUDConverter(
        input_path=Path("tests/data/equipment_maintenance_schedule.xlsx"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


@pytest.mark.slow
class TestImageChunking:
    """We use a class to ensure the order of tests (save then load)"""

    @pytest.fixture(autouse=True)
    def setup(self, output_dir):
        self.output_dir = output_dir
        self.converter = MAUDConverter(
            input_path=Path("tests/data/wiring_bonding.pdf"),
            output_dir=output_dir,
            overwrite=True,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=MAUDPipeline,
                    pipeline_options=MAUDPipelineOptions(
                        generate_page_images=True,
                        generate_picture_images=True,
                        generate_table_images=True,
                        do_page_description=False,
                        images_scale=2.0,
                    ),
                )
            },
        )

    def test_save_images(self):
        self.converter.convert()
        self.converter.save_document()

        # Create directories if they don't exist
        (self.converter._output_path / "pages").mkdir(exist_ok=True)
        (self.converter._output_path / "pictures").mkdir(exist_ok=True)
        (self.converter._output_path / "tables").mkdir(exist_ok=True)

        # Check that images were saved
        assert any(
            (self.converter._output_path / "pages").glob("*.webp")
        ), "No page images found"
        assert any(
            (self.converter._output_path / "pictures").glob("*.webp")
        ), "No picture images found"
        assert any(
            (self.converter._output_path / "tables").glob("*.webp")
        ), "No table images found"

    def test_chunking(self):
        self.converter.convert()
        self.converter.save_document()
        chunks = self.converter.chunk()
        chunk_df = pd.DataFrame(chunks)

        # test for expected columns
        for col in [
            "filename",
            "input_hash",
            "pages",
            "doc_refs",
            "has_table",
            "has_picture",
            "tables",
            "pictures",
            "headings",
            "captions",
            "chunk_type",
            "image_path",
            "text",
            "enriched_text",
            "image_path",
        ]:
            assert col in chunk_df.columns, f"Missing column: {col}"

        # test for image paths
        if not chunk_df.empty:
            page_chunks = chunk_df.query("chunk_type == 'page'")
            if not page_chunks.empty:
                assert (
                    "webp" in page_chunks.iloc[0].image_path
                ), "No webp in page image path"

            picture_chunks = chunk_df.query("chunk_type == 'picture'")
            if not picture_chunks.empty:
                assert (
                    "webp" in picture_chunks.iloc[0].image_path
                ), "No webp in picture image path"

            table_chunks = chunk_df.query("chunk_type == 'table'")
            if not table_chunks.empty:
                assert (
                    "webp" in table_chunks.iloc[0].image_path
                ), "No webp in table image path"

            text_chunks = chunk_df.query("chunk_type == 'text'")
            if not text_chunks.empty:
                assert (
                    text_chunks.iloc[0].image_path == ""
                ), "Text chunk should have empty image path"


class TestDocumentCaching:
    """We use a class to ensure the order of tests (save then load)"""

    @pytest.fixture(autouse=True)
    def setup(self, output_dir):
        self.output_dir = output_dir
        self.converter = MAUDConverter(
            input_path=Path("tests/data/maintenance_procedure_template.docx"),
            output_dir=self.output_dir,
        )

    def test_save_document(self):
        self.converter.convert()
        self.converter.save_document()
        assert next(Path(self.converter._output_path).glob("*.md")).exists()
        assert next(Path(self.converter._output_path).glob("*.json")).exists()

    def test_load_document(self, caplog):
        with caplog.at_level(logging.INFO):
            self.converter.convert()
            assert self.converter.document is not None
            assert "Loading document" in caplog.text

    def test_overwrite_document(self, caplog):
        with caplog.at_level(logging.INFO):
            self.converter = MAUDConverter(
                input_path=Path("tests/data/maintenance_procedure_template.docx"),
                output_dir=self.output_dir,
                overwrite=True,
            )
            self.converter.convert()
            assert "Converting document" in caplog.text
            assert self.converter.document is not None
