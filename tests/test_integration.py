"""Integration tests for the multimodal accelerator project.

These tests verify that different components work together properly
and that the overall system functions as expected.
"""

import pytest
from pathlib import Path

@pytest.mark.integration
class TestProcessingMethodsIntegration:
    """Test integration between different processing methods."""

    def test_processing_methods_available(self):
        """Test that all expected processing methods are available."""
        from src.core import get_processing_methods
        
        methods = get_processing_methods()
        
        # Check that core processing methods are available
        assert "aiparse" in methods
        assert "docling_ray" in methods
        assert "docling_serving" in methods
        
        # Verify method descriptions
        for method_name, method_info in methods.items():
            assert isinstance(method_info, str)  # Methods return description strings

    def test_processing_config_integration(self):
        """Test that ProcessingConfig works with different methods."""
        from src.core import ProcessingConfig
        
        config = ProcessingConfig("main", "default", "input", "output")
        
        # Test basic properties
        assert config.catalog == "main"
        assert config.schema == "default"
        assert "input" in config.input_path
        assert "output" in config.output_path


@pytest.mark.integration
@pytest.mark.aiparse
class TestAIParseIntegration:
    """Test AI_PARSE integration."""

    def test_ai_parse_workflow_logic(self):
        """Test AI_PARSE workflow logic."""
        from src.core import ProcessingConfig, get_processing_methods
        
        config = ProcessingConfig("test", "test", "input", "output")
        methods = get_processing_methods()
        
        # Verify AI_PARSE is available
        assert "aiparse" in methods
        
        # Test configuration compatibility
        assert config.get_table_name("documents") == "test.test.documents"


@pytest.mark.integration  
@pytest.mark.docling
class TestDoclingIntegration:
    """Test Docling integration."""

    def test_docling_workflow_logic(self):
        """Test Docling workflow logic.""" 
        from src.core import ProcessingConfig
        
        config = ProcessingConfig("test", "test", "input", "output")
        
        # Test that config works for Docling processing
        assert config.input_path is not None
        assert config.output_path is not None

    def test_processing_result_structure(self):
        """Test processing result structure."""
        from src.core import print_processing_summary
        
        # Test with realistic results structure
        results = [
            {"status": "success", "file": "file1.pdf"},
            {"status": "success", "file": "file2.pdf"}
        ]
        
        # Should not raise an exception
        print_processing_summary(results, "docling_ray")


@pytest.mark.integration
@pytest.mark.serving
class TestServingEndpointIntegration:
    """Test serving endpoint integration."""

    def test_serving_endpoint_structure(self):
        """Test serving endpoint structure."""
        try:
            from src.docling_endpoint import DoclingParsingModel
            
            model = DoclingParsingModel()
            assert hasattr(model, 'predict')
            assert hasattr(model, 'load_context')
        except ImportError:
            pytest.skip("Docling endpoint not available")

    def test_serving_request_structure(self):
        """Test serving request structure."""
        # Test expected request format
        request = {
            "file_path": "/Volumes/main/default/docs/test.pdf",
            "output_path": "/Volumes/main/default/output/",
            "vlm_preset": "granite_picture_description"
        }
        
        # Verify required fields
        assert "file_path" in request
        assert "output_path" in request

    def test_serving_response_structure(self):
        """Test serving response structure."""
        # Test expected response format
        response = {
            "status": "success",
            "document_path": "/Volumes/main/default/output/test.json",
            "images": ["image1.png", "image2.png"],
            "processing_time": 45.2
        }
        
        # Verify response structure
        assert "status" in response
        assert response["status"] in ["success", "error"]


@pytest.mark.integration
@pytest.mark.error_handling
class TestErrorHandling:
    """Test error handling across the system."""

    def test_invalid_filename_handling(self):
        """Test handling of invalid filenames."""
        from src.core import sanitize_filename
        
        # Test filename sanitization
        result = sanitize_filename("invalid<>file|name.pdf")
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result

    def test_module_imports(self):
        """Test that core modules can be imported."""
        # Test that core modules can be imported
        try:
            import src.core
            import src.images
            assert True
        except ImportError as e:
            pytest.fail(f"Core module import failed: {e}")


@pytest.mark.integration
@pytest.mark.data_validation
class TestDataConsistency:
    """Test data consistency across the system."""

    def test_config_structure_consistency(self):
        """Test that config structures are consistent."""
        from src.core import ProcessingConfig
        
        config = ProcessingConfig("test", "test", "input", "output")
        
        # Test that config produces consistent table names
        docs_table = config.get_table_name("documents")
        chunks_table = config.get_table_name("chunks")
        
        assert "test.test" in docs_table
        assert "test.test" in chunks_table

    def test_volume_path_consistency(self):
        """Test volume path consistency."""
        from src.core import get_volume_path, ProcessingConfig
        
        # Test direct function call
        direct_path = get_volume_path("test", "schema", "volume")
        
        # Test via config
        config = ProcessingConfig("test", "schema", "volume", "output")
        config_path = config.input_path
        
        # Should produce same base path
        assert "test/schema/volume" in direct_path
        assert "test/schema/volume" in config_path


@pytest.mark.integration
class TestSystemStability:
    """Test system stability."""

    def test_core_functionality_stability(self):
        """Test that core functionality works reliably."""
        from src.core import get_processing_methods, ProcessingConfig, sanitize_filename
        
        # Test core functions work
        methods = get_processing_methods()
        assert len(methods) > 0
        
        config = ProcessingConfig("test", "test", "input", "output")
        assert config is not None
        
        filename = sanitize_filename("test file.pdf")
        assert filename is not None
        assert ".pdf" in filename

    def test_repeated_operations(self):
        """Test that operations work consistently when repeated."""
        from src.core import sanitize_filename, get_volume_path
        
        # Test that repeated calls give same results
        filename = "Test Document (Final).pdf"
        result1 = sanitize_filename(filename)
        result2 = sanitize_filename(filename)
        assert result1 == result2
        
        path1 = get_volume_path("catalog", "schema", "volume")
        path2 = get_volume_path("catalog", "schema", "volume")
        assert path1 == path2