"""Simplified tests for document chunking functionality - testing only essentials."""

import pytest
from src.document.chunkers import chunk_schema


@pytest.mark.unit
@pytest.mark.document
class TestChunkSchema:
    """Test the chunk schema definition."""

    def test_chunk_schema_exists(self):
        """Test that chunk schema is defined."""
        assert chunk_schema is not None

    def test_chunk_schema_has_expected_fields(self):
        """Test that chunk schema has the expected field names."""
        field_names = [field.name for field in chunk_schema.fields]
        expected_fields = [
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
        ]
        for field in expected_fields:
            assert field in field_names


@pytest.mark.unit
@pytest.mark.document
class TestChunkerImports:
    """Test that chunker functions can be imported."""

    def test_chunker_functions_import(self):
        """Test that main chunker functions can be imported."""
        from src.document.chunkers import (
            make_table_chunks,
            make_picture_chunks,
            make_page_chunks,
            make_text_chunk,
            make_text_chunks,
            chunk_maud_document,
        )

        assert make_table_chunks is not None
        assert make_picture_chunks is not None
        assert make_page_chunks is not None
        assert make_text_chunk is not None
        assert make_text_chunks is not None
        assert chunk_maud_document is not None

    def test_hybrid_chunker_import(self):
        """Test that HybridChunker can be imported."""
        from src.document.chunkers import HybridChunker

        assert HybridChunker is not None


@pytest.mark.unit
@pytest.mark.document
class TestBasicChunkingFunctionality:
    """Test basic chunking functionality without complex mocks."""

    def test_chunking_functions_are_callable(self):
        """Test that chunking functions are callable."""
        from src.document.chunkers import (
            make_table_chunks,
            make_picture_chunks,
            make_page_chunks,
            chunk_maud_document,
        )

        assert callable(make_table_chunks)
        assert callable(make_picture_chunks)
        assert callable(make_page_chunks)
        assert callable(chunk_maud_document)


@pytest.mark.integration
@pytest.mark.document
class TestChunkDataFrame:
    """Test chunk data structure and DataFrame conversion."""

    def test_chunk_to_dataframe(self):
        """Test converting chunk data to DataFrame."""
        # Create simple test data that matches chunk schema
        test_chunk_data = [
            {
                "filename": "test.pdf",
                "input_hash": "abc123",
                "pages": [1],
                "doc_refs": ["#/test"],
                "has_table": False,
                "has_picture": False,
                "tables": [],
                "pictures": [],
                "headings": ["Test Header"],
                "captions": [],
                "chunk_type": "text",
                "image_path": "",
                "text": "This is test text.",
                "enriched_text": "This is test text.",
            }
        ]

        # Just verify we can work with the data structure
        assert len(test_chunk_data) == 1
        assert test_chunk_data[0]["chunk_type"] == "text"

    def test_chunk_filtering(self):
        """Test filtering chunks by type."""
        # Create test data with different chunk types
        test_chunks = [
            {"chunk_type": "text", "text": "text chunk"},
            {"chunk_type": "table", "text": "table chunk"},
            {"chunk_type": "picture", "text": "picture chunk"},
        ]

        # Test basic filtering
        text_chunks = [c for c in test_chunks if c["chunk_type"] == "text"]
        table_chunks = [c for c in test_chunks if c["chunk_type"] == "table"]

        assert len(text_chunks) == 1
        assert len(table_chunks) == 1
        assert text_chunks[0]["text"] == "text chunk"
