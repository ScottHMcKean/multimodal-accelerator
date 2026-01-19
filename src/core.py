"""
Core utilities for the three document processing methods.

This module provides the essential shared functionality for:
1. AI_PARSE - Databricks native parsing
2. Docling + Ray - Parallel processing  
3. Docling Serving - GPU-accelerated endpoint
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_volume_path(catalog: str, schema: str, volume: str) -> str:
    """Get standardized volume path."""
    return f"/Volumes/{catalog}/{schema}/{volume}"


def get_file_list(directory: Union[str, Path], pattern: str = "*.pdf") -> List[Path]:
    """Get list of files matching pattern in directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    files = list(dir_path.rglob(pattern))
    logger.info(f"Found {len(files)} files matching {pattern} in {directory}")
    return files


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    import re
    
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    
    # Replace URL encoding and clean up
    stem = stem.replace('%20', '_')
    sanitized = re.sub(r'[^\w\s-]', '', stem)
    sanitized = sanitized.replace(' ', '_')
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_').lower()
    
    return f"{sanitized}{suffix}"


def print_processing_summary(results: List[Dict], method_name: str):
    """Print standardized processing summary."""
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"{method_name.upper()} PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        success_rate = (successful / len(results)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Show failed files (first 5)
    failed_results = [r for r in results if r.get("status") == "error"]
    if failed_results:
        print(f"\n❌ Failed files:")
        for result in failed_results[:5]:
            file_name = Path(result.get("file", "unknown")).name
            error = result.get("error", "Unknown error")
            print(f"  {file_name}: {error}")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more")


class ProcessingConfig:
    """Configuration for document processing methods."""
    
    def __init__(
        self,
        catalog: str = "main",
        schema: str = "default", 
        input_volume: str = "raw_docs",
        output_volume: str = "processed_docs"
    ):
        self.catalog = catalog
        self.schema = schema
        self.input_volume = input_volume
        self.output_volume = output_volume
    
    @property
    def input_path(self) -> str:
        return get_volume_path(self.catalog, self.schema, self.input_volume)
    
    @property 
    def output_path(self) -> str:
        return get_volume_path(self.catalog, self.schema, self.output_volume)
    
    def get_table_name(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"


# Standard configuration
DEFAULT_CONFIG = ProcessingConfig()


def get_processing_methods() -> Dict[str, str]:
    """Get available processing methods and their descriptions."""
    return {
        "aiparse": "Databricks native ai_parse() function - serverless, simple",
        "docling_ray": "Docling with Ray parallel processing - scalable, flexible", 
        "docling_serving": "Docling model serving endpoint - GPU-accelerated, VLM support"
    }


def print_method_comparison():
    """Print comparison of the three processing methods."""
    print("📊 Document Processing Method Comparison")
    print("=" * 60)
    
    methods = [
        {
            "name": "AI_PARSE",
            "complexity": "Simple",
            "setup": "None",
            "scaling": "Automatic", 
            "vlm": "No",
            "best_for": "Large batch processing, serverless"
        },
        {
            "name": "Docling + Ray", 
            "complexity": "Medium",
            "setup": "Ray cluster",
            "scaling": "Manual",
            "vlm": "Optional",
            "best_for": "Flexible parallel processing"
        },
        {
            "name": "Docling Serving",
            "complexity": "Low",
            "setup": "Model endpoint", 
            "scaling": "Automatic",
            "vlm": "Native",
            "best_for": "GPU acceleration, VLM descriptions"
        }
    ]
    
    # Print table header
    print(f"{'Method':<15} {'Complexity':<10} {'Setup':<12} {'Scaling':<10} {'VLM':<8} {'Best For'}")
    print("-" * 80)
    
    # Print table rows
    for method in methods:
        print(f"{method['name']:<15} {method['complexity']:<10} {method['setup']:<12} "
              f"{method['scaling']:<10} {method['vlm']:<8} {method['best_for']}")
    
    print("\n💡 Choose based on your specific needs:")
    print("  • AI_PARSE: Simplest, works with serverless")
    print("  • Docling + Ray: Most flexible, custom processing")  
    print("  • Docling Serving: Most advanced, GPU + VLM support")


if __name__ == "__main__":
    print_method_comparison()