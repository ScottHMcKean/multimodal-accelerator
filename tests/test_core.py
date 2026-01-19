"""Tests for src/core.py - Core utility functions."""

import pytest
from pathlib import Path
import tempfile
import os

from src.core import (
    get_volume_path,
    sanitize_filename,
    ProcessingConfig,
    get_processing_methods,
    print_processing_summary,
)


class TestVolumeUtilities:
    """Test volume path utilities."""

    def test_get_volume_path(self):
        """Test volume path generation."""
        path = get_volume_path("main", "default", "docs")
        assert path == "/Volumes/main/default/docs"

    def test_get_volume_path_with_special_chars(self):
        """Test volume path with special characters."""
        path = get_volume_path("test-catalog", "schema_name", "volume.name")
        assert path == "/Volumes/test-catalog/schema_name/volume.name"


class TestFilenameUtilities:
    """Test filename utilities."""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        result = sanitize_filename("normal_file.pdf")
        assert result == "normal_file.pdf"

    def test_sanitize_filename_url_encoding(self):
        """Test URL-encoded filename sanitization."""
        result = sanitize_filename("My%20Document.pdf")
        assert result == "my_document.pdf"

    def test_sanitize_filename_special_chars(self):
        """Test special character sanitization."""
        result = sanitize_filename("file<>name|with:bad*chars?.pdf")
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
        assert result.endswith(".pdf")

    def test_sanitize_filename_multiple_underscores(self):
        """Test multiple underscores are collapsed."""
        result = sanitize_filename("file___with___multiple___underscores.pdf")
        assert "___" not in result
        assert result.endswith(".pdf")

    def test_sanitize_filename_preserve_extension(self):
        """Test extension preservation."""
        result = sanitize_filename("document.docx")
        assert result.endswith(".docx")


class TestProcessingConfig:
    """Test ProcessingConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProcessingConfig("main", "default", "input", "output")

        assert config.catalog == "main"
        assert config.schema == "default"
        assert config.input_volume == "input"
        assert config.output_volume == "output"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProcessingConfig("test", "schema", "docs", "results")

        assert config.catalog == "test"
        assert config.schema == "schema"
        assert config.input_volume == "docs"
        assert config.output_volume == "results"

    def test_input_path_property(self):
        """Test input path property."""
        config = ProcessingConfig("main", "default", "input", "output")
        expected = "/Volumes/main/default/input"
        assert config.input_path == expected

    def test_output_path_property(self):
        """Test output path property."""
        config = ProcessingConfig("main", "default", "input", "output")
        expected = "/Volumes/main/default/output"
        assert config.output_path == expected

    def test_get_table_name(self):
        """Test table name generation."""
        config = ProcessingConfig("test", "schema", "input", "output")

        docs_table = config.get_table_name("documents")
        assert docs_table == "test.schema.documents"

        chunks_table = config.get_table_name("chunks")
        assert chunks_table == "test.schema.chunks"


class TestProcessingMethods:
    """Test processing method utilities."""

    def test_get_processing_methods(self):
        """Test processing methods retrieval."""
        methods = get_processing_methods()

        assert isinstance(methods, dict)
        assert len(methods) > 0

        # Check that all three core methods are available
        assert "aiparse" in methods
        assert "docling_ray" in methods
        assert "docling_serving" in methods

        # Check that values are descriptions
        for method_name, description in methods.items():
            assert isinstance(description, str)
            assert len(description) > 0


class TestProcessingSummary:
    """Test processing summary functionality."""

    def test_print_processing_summary_success(self, capsys):
        """Test processing summary with all successes."""
        results = [
            {"status": "success", "file": "doc1.pdf"},
            {"status": "success", "file": "doc2.pdf"},
        ]

        print_processing_summary(results, "aiparse")
        captured = capsys.readouterr()

        assert "PROCESSING SUMMARY" in captured.out
        assert "AIPARSE" in captured.out
        assert "2" in captured.out  # Total processed

    def test_print_processing_summary_mixed(self, capsys):
        """Test processing summary with mixed results."""
        results = [
            {"status": "success", "file": "doc1.pdf"},
            {"status": "error", "file": "doc2.pdf", "error": "Failed to parse"},
        ]

        print_processing_summary(results, "docling_ray")
        captured = capsys.readouterr()

        assert "PROCESSING SUMMARY" in captured.out
        assert "DOCLING_RAY" in captured.out

    def test_print_processing_summary_empty_results(self, capsys):
        """Test processing summary with empty results."""
        results = []

        print_processing_summary(results, "docling_serving")
        captured = capsys.readouterr()

        assert "PROCESSING SUMMARY" in captured.out


class TestIntegration:
    """Test integration of core components."""

    def test_full_processing_config_workflow(self):
        """Test complete processing configuration workflow."""
        # Create config
        config = ProcessingConfig("test", "default", "documents", "parsed")

        # Test paths
        assert config.input_path == "/Volumes/test/default/documents"
        assert config.output_path == "/Volumes/test/default/parsed"

        # Test table names
        docs_table = config.get_table_name("documents")
        chunks_table = config.get_table_name("chunks")

        assert docs_table == "test.default.documents"
        assert chunks_table == "test.default.chunks"

    def test_filename_sanitization_workflow(self):
        """Test filename sanitization workflow."""
        filenames = [
            "My Document (Final).pdf",
            "Report-2024@Company.docx",
            "Data Analysis Results.xlsx",
        ]

        sanitized = [sanitize_filename(f) for f in filenames]

        expected = [
            "my_document_final.pdf",  # Parentheses are replaced with underscores, lowercase
            "report-2024company.docx",
            "data_analysis_results.xlsx",
        ]

        assert sanitized == expected

    def test_processing_summary_with_realistic_data(self, capsys):
        """Test processing summary with realistic data."""
        results = [
            {"status": "success", "file": "equipment_maintenance.pdf", "pages": 15},
            {"status": "success", "file": "safety_procedures.docx", "pages": 8},
            {
                "status": "error",
                "file": "corrupted_file.pdf",
                "error": "Invalid PDF format",
            },
        ]

        print_processing_summary(results, "aiparse")
        captured = capsys.readouterr()

        # Verify key information is present
        assert "PROCESSING SUMMARY" in captured.out
        assert "AIPARSE" in captured.out
        assert "3" in captured.out or "Total" in captured.out
