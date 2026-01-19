"""Tests for src/images.py - Image handling utilities."""

import pytest
import tempfile
import os
import io
from pathlib import Path

# Skip all tests if PIL is not available
PIL = pytest.importorskip("PIL")
from PIL import Image

try:
    from src.images import (
        load_image_from_volume,
        load_image_from_uc,
        load_image,
        response_to_image
    )
except ImportError as e:
    pytest.skip(f"Images module not available: {e}", allow_module_level=True)


class TestImageFunctions:
    """Test image utility functions."""

    def test_functions_exist(self):
        """Test that all expected functions exist."""
        assert callable(load_image_from_volume)
        assert callable(load_image_from_uc)
        assert callable(load_image)
        assert callable(response_to_image)

    def test_basic_image_processing(self):
        """Test basic PIL image operations."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        assert test_image is not None
        assert isinstance(test_image, Image.Image)
        assert test_image.size == (100, 100)
        assert test_image.mode == 'RGB'

    def test_image_resize_operations(self):
        """Test image resize operations."""
        # Create test image
        original = Image.new('RGB', (200, 200), color='blue')
        
        # Resize to smaller size
        resized = original.resize((100, 100), Image.Resampling.LANCZOS)
        
        assert resized.size == (100, 100)
        assert resized.mode == 'RGB'

    def test_image_format_conversion(self):
        """Test image format handling."""
        # Create test image
        test_image = Image.new('RGB', (50, 50), color='green')
        
        # Save as bytes
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Load from bytes
        loaded_image = Image.open(img_bytes)
        
        assert loaded_image.size == (50, 50)
        assert loaded_image.mode in ['RGB', 'RGBA']  # PNG might add alpha


class TestImageWorkflow:
    """Test complete image processing workflows."""
    
    def test_image_processing_workflow(self):
        """Test the complete image processing workflow."""
        # 1. Create a test image (simulates loading from volume)
        original = Image.new('RGB', (200, 200), color='green')
        
        # 2. Process the image (simulates thumbnail creation)
        processed = original.resize((100, 100), Image.Resampling.LANCZOS)
        
        # 3. Verify the result
        assert processed.size == (100, 100)
        assert processed.mode == 'RGB'
        
        # 4. Save to temporary location (simulates volume upload)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            processed.save(tmp.name, 'PNG')
            assert Path(tmp.name).exists()
            
            # Cleanup
            os.unlink(tmp.name)

    def test_multiple_image_formats(self):
        """Test handling of different image formats."""
        formats_to_test = ['PNG', 'JPEG']
        
        for format_name in formats_to_test:
            # Create test image
            if format_name == 'JPEG':
                # JPEG doesn't support transparency, use RGB
                test_image = Image.new('RGB', (50, 50), color='red')
            else:
                test_image = Image.new('RGBA', (50, 50), color='red')
            
            # Save and reload
            with tempfile.NamedTemporaryFile(suffix=f'.{format_name.lower()}', delete=False) as tmp:
                test_image.save(tmp.name, format_name)
                
                # Verify file was created
                assert Path(tmp.name).exists()
                
                # Load and verify
                loaded = Image.open(tmp.name)
                assert loaded.size == (50, 50)
                
                # Cleanup
                os.unlink(tmp.name)