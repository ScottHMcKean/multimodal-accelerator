from maud.document.converters import MaudConverter
from pathlib import Path
import pytest
import shutil


def test_converter_instantiation():
    converter = MaudConverter(
        input_path=Path("tests/data/wind_turbine.pdf"),
        output_dir=Path("tests/data/output"),
    )
    assert isinstance(converter, MaudConverter)
    assert isinstance(converter.input_path, Path)
    assert isinstance(converter.output_dir, Path)


@pytest.mark.slow
def test_convert_pdf():
    converter = MaudConverter(
        input_path=Path("tests/data/wind_turbine.pdf"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


def test_convert_docx():
    converter = MaudConverter(
        input_path=Path("tests/data/maintenance_procedure_template.docx"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


def test_convert_pptx():
    converter = MaudConverter(
        input_path=Path("tests/data/functional_flight_checks.pptx"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


def test_convert_xlsx():
    converter = MaudConverter(
        input_path=Path("tests/data/equipment_maintenance_schedule.xlsx"),
        output_dir=Path("tests/data/output"),
    )
    converter.convert()
    assert converter.document is not None


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory and clean it up after tests."""
    path = Path("tests/data/output")
    path.mkdir(exist_ok=True)
    yield path
    shutil.rmtree(path)


class TestDocumentCaching:
    """We use a class to ensure the order of tests (save then load)"""

    @pytest.fixture(autouse=True)
    def setup(self, output_dir):
        self.output_dir = output_dir
        self.converter = MaudConverter(
            input_path=Path("tests/data/maintenance_procedure_template.docx"),
            output_dir=self.output_dir,
        )

    def test_save_document(self):
        self.converter.convert()
        self.converter.save_document()
        assert next(Path(self.converter._output_path).glob("*.md")).exists()

    def test_load_document(self, caplog):
        self.converter.convert()
        assert self.converter.document is not None
        assert "Loading document" in caplog.text

    def test_overwrite_document(self, caplog):
        self.converter = MaudConverter(
            input_path=Path("tests/data/maintenance_procedure_template.docx"),
            output_dir=self.output_dir,
            overwrite=True,
        )
        self.converter.convert()
        assert "Converting document" in caplog.text
        assert self.converter.document is not None
