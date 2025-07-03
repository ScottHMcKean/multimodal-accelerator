"""Simplified integration tests for essential MAUD functionality."""

import pytest
from pathlib import Path


@pytest.mark.integration
@pytest.mark.document
class TestBasicDocumentProcessing:
    """Test basic document processing functionality."""

    def test_document_converter_import(self):
        """Test that document converter can be imported."""
        from src.document.converters import MAUDConverter

        assert MAUDConverter is not None

    def test_document_processing_with_sample_files(self, sample_docx_path):
        """Test basic document processing with sample files."""
        from src.document.converters import MAUDConverter
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            converter = MAUDConverter(
                input_path=sample_docx_path, output_dir=Path(temp_dir)
            )

            # Just test that converter can be created
            assert converter is not None
            assert converter.input_path.exists()

    def test_different_file_types_basic(
        self, sample_docx_path, sample_xlsx_path, sample_pptx_path
    ):
        """Test that converters can be created for different file types."""
        from src.document.converters import MAUDConverter
        from tempfile import TemporaryDirectory

        file_paths = [sample_docx_path, sample_xlsx_path, sample_pptx_path]

        with TemporaryDirectory() as temp_dir:
            for file_path in file_paths:
                if file_path.exists():
                    converter = MAUDConverter(
                        input_path=file_path, output_dir=Path(temp_dir)
                    )
                    assert converter is not None


@pytest.mark.integration
@pytest.mark.document
class TestModuleIntegration:
    """Test that modules work together."""

    def test_agent_document_module_integration(self):
        """Test that agent and document modules can be imported together."""
        from src.agent import config, functions, nodes
        from src.document import converters, chunkers

        # Test that modules don't conflict
        assert config is not None
        assert functions is not None
        assert nodes is not None
        assert converters is not None
        assert chunkers is not None

    def test_interface_module_integration(self):
        """Test that interface modules can be imported."""
        from src.interface import config

        assert config is not None


@pytest.mark.integration
@pytest.mark.error_handling
class TestBasicErrorHandling:
    """Test basic error handling."""

    def test_invalid_file_handling(self):
        """Test handling of invalid input files."""
        from src.document.converters import MAUDConverter
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            invalid_path = Path("nonexistent_file.pdf")

            # Should handle invalid path gracefully
            try:
                converter = MAUDConverter(
                    input_path=invalid_path, output_dir=Path(temp_dir)
                )
                # Just test creation, not conversion
                assert converter is not None
            except Exception as e:
                # If it raises an exception, that's also acceptable behavior
                assert isinstance(e, (FileNotFoundError, ValueError))

    def test_module_import_error_handling(self):
        """Test that module imports handle missing dependencies gracefully."""
        # Test that basic imports work
        try:
            from src.agent import config

            assert config is not None
        except ImportError:
            pytest.skip("Agent config module not available")

        try:
            from src.document import converters

            assert converters is not None
        except ImportError:
            pytest.skip("Document converters module not available")


@pytest.mark.integration
@pytest.mark.data_validation
class TestDataConsistency:
    """Test data consistency across components."""

    def test_chunk_data_structure_consistency(self):
        """Test that chunk data structure is consistent."""
        from src.document.chunkers import chunk_schema

        # Test that schema is consistently defined
        assert chunk_schema is not None
        field_names = [field.name for field in chunk_schema.fields]
        assert len(field_names) > 0
        assert "chunk_type" in field_names
        assert "text" in field_names

    def test_config_structure_consistency(self):
        """Test that config structures are consistent."""
        from src.agent.config import MaudConfig

        # Test that config classes exist and are properly structured
        assert MaudConfig is not None
        # Test that it's a class that can be referenced
        assert hasattr(MaudConfig, "__name__")


@pytest.mark.regression
@pytest.mark.integration
class TestSystemStability:
    """Basic regression tests for system stability."""

    def test_module_import_stability(self):
        """Test that core modules can be imported consistently."""
        # Test multiple imports of the same modules
        for _ in range(3):
            from src.agent import config
            from src.document import chunkers

            assert config is not None
            assert chunkers is not None

    def test_basic_functionality_stability(self):
        """Test that basic functionality remains stable."""
        from src.agent.functions import add
        from src.document.chunkers import chunk_schema

        # Test that basic functions work consistently
        assert add(2, 3) == 5
        assert add(0, 0) == 0
        assert add(-1, 1) == 0

        # Test that schema remains consistent
        assert chunk_schema is not None
        field_count = len(chunk_schema.fields)
        assert field_count > 10  # Should have many fields
