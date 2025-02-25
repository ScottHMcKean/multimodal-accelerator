from maud.document.converters import MAUDConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from maud.document.converters import MAUDPipelineOptions, MAUDConverter, MAUDPipeline
from pathlib import Path
import pytest
import shutil
import pandas as pd


@pytest.fixture(scope="session")
def output_dir():
    """Create output directory and clean it up after tests."""
    path = Path("tests/data/output")
    path.mkdir(exist_ok=True)
    yield path
    shutil.rmtree(path)


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
                    pipeline_options=MAUDPipelineOptions(),
                )
            },
        )

    def test_save_images(self):
        self.converter.convert()
        self.converter.save_document()
        assert next((self.converter._output_path / "pages").glob("*.webp")).exists()
        assert next((self.converter._output_path / "pictures").glob("*.webp")).exists()
        assert next((self.converter._output_path / "tables").glob("*.webp")).exists()

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
        (chunk_df)
        assert "webp" in chunk_df.query("chunk_type == 'page'").iloc[0].image_path
        assert "webp" in chunk_df.query("chunk_type == 'picture'").iloc[0].image_path
        assert "webp" in chunk_df.query("chunk_type == 'table'").iloc[0].image_path
        assert chunk_df.query("chunk_type == 'text'").iloc[0].image_path == ""


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
        self.converter.convert()
        assert self.converter.document is not None
        assert "Loading document" in caplog.text

    def test_overwrite_document(self, caplog):
        self.converter = MAUDConverter(
            input_path=Path("tests/data/maintenance_procedure_template.docx"),
            output_dir=self.output_dir,
            overwrite=True,
        )
        self.converter.convert()
        assert "Converting document" in caplog.text
        assert self.converter.document is not None
