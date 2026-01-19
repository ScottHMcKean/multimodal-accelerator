"""
Custom Model Serving Endpoint for Docling Document Parsing

This module provides a custom MLflow model serving endpoint that:
1. Accepts file paths from Databricks Volumes
2. Downloads files using the Volume SDK
3. Parses documents using Docling
4. Uploads parsed JSON and extracted images back to the volume
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.files import DownloadResponse
from mlflow.pyfunc import PythonModel
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode

from src.core import sanitize_filename


class DoclingParsingModel(PythonModel):
    """
    Custom MLflow model for document parsing using Docling.

    This model accepts file paths, downloads files from Databricks Volumes,
    parses them with Docling, and uploads results back to the volume.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_client = None
        self.converter = None

    def load_context(self, context):
        """Initialize the model with necessary components."""
        self.logger.info("Loading Docling parsing model context")

        # Initialize Databricks workspace client
        self.workspace_client = WorkspaceClient()

        # We'll initialize SimpleDocumentConverter per request with dynamic options
        # This allows each request to specify different VLM settings

        # Set up temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger.info(f"Using temporary directory: {self.temp_dir}")

    def predict(self, context, model_input):
        """
        Parse document from volume path with native Docling VLM support.

        Args:
            model_input: Dictionary containing:
                - file_path: Path to file in Databricks Volume
                - output_path: Path where to store results in Volume
                - options: Optional parsing options including:
                    # Core options:
                    - images_scale: float (default 2.0)
                    - generate_page_images: bool (default True)
                    - do_ocr: bool (default True)

                    # VLM Picture Descriptions (NEW!):
                    - vlm_preset: "granite" or "smolvlm"
                    - vlm_repo_id: Custom HuggingFace model repo
                    - vlm_prompt: Custom prompt for descriptions
                    - do_picture_description: Enable VLM descriptions

        Returns:
            Dictionary with parsing results, VLM descriptions, and output locations
        """
        try:
            # Handle MLflow's list input format - take first item
            if isinstance(model_input, list):
                input_data = model_input[0] if model_input else {}
            else:
                input_data = model_input

            # Extract input parameters
            file_path = input_data.get("file_path")
            output_path = input_data.get("output_path")
            options = input_data.get("options", {})

            if not file_path:
                raise ValueError("file_path is required")
            if not output_path:
                raise ValueError("output_path is required")

            self.logger.info(f"Processing file: {file_path}")

            # Download file from volume
            local_file_path = self._download_file_from_volume(file_path)

            # Set up local output directory
            local_output_dir = (
                self.temp_dir / "output" / sanitize_filename(Path(file_path).stem)
            )
            local_output_dir.mkdir(parents=True, exist_ok=True)

            # Parse document using native Docling with VLM support
            self.logger.info("Starting document parsing with native Docling")

            # Set up Docling converter with VLM options
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                granite_picture_description,
                smolvlm_picture_description,
                PictureDescriptionVlmOptions,
            )

            # Configure pipeline options
            pipeline_options = PdfPipelineOptions(
                images_scale=options.get("images_scale", 2.0),
                generate_page_images=options.get("generate_page_images", True),
                generate_picture_images=options.get("generate_picture_images", True),
                generate_table_images=options.get("generate_table_images", True),
                do_ocr=options.get("do_ocr", True),
                do_table_structure=options.get("do_table_structure", True),
            )

            # Configure VLM if requested
            vlm_preset = options.get("vlm_preset")
            vlm_repo_id = options.get("vlm_repo_id")
            vlm_prompt = options.get("vlm_prompt", "Describe this image in detail.")

            if vlm_preset == "granite":
                pipeline_options.do_picture_description = True
                pipeline_options.picture_description_options = (
                    granite_picture_description
                )
                pipeline_options.picture_description_options.prompt = vlm_prompt
            elif vlm_preset == "smolvlm":
                pipeline_options.do_picture_description = True
                pipeline_options.picture_description_options = (
                    smolvlm_picture_description
                )
                pipeline_options.picture_description_options.prompt = vlm_prompt
            elif vlm_repo_id:
                pipeline_options.do_picture_description = True
                pipeline_options.picture_description_options = (
                    PictureDescriptionVlmOptions(
                        repo_id=vlm_repo_id,
                        prompt=vlm_prompt,
                    )
                )

            # Initialize converter
            if pipeline_options.do_picture_description:
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                        )
                    }
                )
            else:
                converter = DocumentConverter()

            # Convert document
            result = converter.convert(source=local_file_path)
            document = result.document

            # Save parsed results locally
            self._save_document_locally(document, local_output_dir)

            # Upload results back to volume
            upload_results = self._upload_results_to_volume(
                local_output_dir=local_output_dir, volume_output_path=output_path
            )

            # Get document info
            doc_info = {
                "pages": len(document.pages),
                "pictures": len(document.pictures),
                "tables": len(document.tables),
                "main_text_length": (
                    len(document.main_text) if hasattr(document, "main_text") else 0
                ),
                "vlm_enabled": bool(vlm_preset or vlm_repo_id),
            }

            # Get VLM-generated descriptions if available
            picture_descriptions = {}
            if pipeline_options.do_picture_description:
                for i, picture in enumerate(document.pictures):
                    if hasattr(picture, "annotations") and picture.annotations:
                        for annotation in picture.annotations:
                            if hasattr(annotation, "text") and annotation.text:
                                picture_descriptions[i] = annotation.text
                                break
                doc_info["pictures_with_descriptions"] = len(picture_descriptions)
            else:
                doc_info["pictures_with_descriptions"] = 0

            # Cleanup local files
            self._cleanup_local_files(local_file_path, local_output_dir)

            # Return success response (wrapped in list for MLflow)
            result = {
                "status": "success",
                "input_file": file_path,
                "output_location": output_path,
                "uploaded_files": upload_results,
                "document_info": doc_info,
                "picture_descriptions": picture_descriptions,
                "processing_options": options,
            }
            return [result]

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            # Return error response (wrapped in list for MLflow)
            error_result = {
                "status": "error",
                "error": str(e),
                "input_file": input_data.get("file_path", "unknown"),
            }
            return [error_result]

    def _download_file_from_volume(self, volume_path: str) -> Path:
        """Download file from Databricks Volume to local temporary storage."""
        self.logger.info(f"Downloading file from volume: {volume_path}")

        # Create local file path
        filename = Path(volume_path).name
        local_path = self.temp_dir / "input" / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download file using Databricks Files API
            with self.workspace_client.files.download(volume_path) as download_response:
                with open(local_path, "wb") as local_file:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        if chunk:
                            local_file.write(chunk)

            self.logger.info(f"Successfully downloaded to: {local_path}")
            return local_path

        except Exception as e:
            self.logger.error(f"Failed to download file {volume_path}: {str(e)}")
            raise

    def _save_document_locally(self, document, output_dir: Path):
        """Save document and images to local directory."""
        self.logger.info(f"Saving document locally to: {output_dir}")

        try:
            # Save document as JSON
            json_file = output_dir / "doc.json"
            document.save_as_json(json_file, image_mode=ImageRefMode.EMBEDDED)

            # Save document as Markdown
            md_file = output_dir / "doc.md"
            document.save_as_markdown(md_file, image_mode=ImageRefMode.EMBEDDED)

            # Save images in separate directories
            self._save_document_images(document, output_dir)

            self.logger.info("Document saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save document locally: {str(e)}")
            raise

    def _save_document_images(self, document, output_dir: Path):
        """Save document images to separate directories."""
        directories = {
            "pages": output_dir / "pages",
            "pictures": output_dir / "pictures",
            "tables": output_dir / "tables",
        }

        # Create directories
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save page images
        for page_no, page in document.pages.items():
            if hasattr(page, "image") and page.image is not None:
                try:
                    page_image_path = directories["pages"] / f"page_{page_no}.webp"
                    page.image.pil_image.save(
                        page_image_path, format="webp", quality=85
                    )
                    self.logger.debug(f"Saved page image: {page_image_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save page {page_no} image: {e}")

        # Save picture images
        for i, picture in enumerate(document.pictures):
            if hasattr(picture, "image") and picture.image is not None:
                try:
                    pic_image_path = directories["pictures"] / f"picture_{i}.webp"
                    picture.image.pil_image.save(
                        pic_image_path, format="webp", quality=85
                    )
                    self.logger.debug(f"Saved picture image: {pic_image_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save picture {i} image: {e}")

        # Save table images
        for i, table in enumerate(document.tables):
            if hasattr(table, "image") and table.image is not None:
                try:
                    table_image_path = directories["tables"] / f"table_{i}.webp"
                    table.image.pil_image.save(
                        table_image_path, format="webp", quality=85
                    )
                    self.logger.debug(f"Saved table image: {table_image_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save table {i} image: {e}")

    def _upload_results_to_volume(
        self, local_output_dir: Path, volume_output_path: str
    ) -> Dict[str, List[str]]:
        """Upload parsed results to Databricks Volume."""
        self.logger.info(f"Uploading results to volume: {volume_output_path}")

        uploaded_files = {"json": [], "markdown": [], "images": []}

        try:
            # Upload main document files
            for file_path in local_output_dir.iterdir():
                if file_path.is_file():
                    volume_file_path = (
                        f"{volume_output_path.rstrip('/')}/{file_path.name}"
                    )

                    with open(file_path, "rb") as local_file:
                        self.workspace_client.files.upload(
                            file_path=volume_file_path,
                            contents=local_file,
                            overwrite=True,
                        )

                    if file_path.suffix == ".json":
                        uploaded_files["json"].append(volume_file_path)
                    elif file_path.suffix == ".md":
                        uploaded_files["markdown"].append(volume_file_path)

                    self.logger.info(f"Uploaded: {volume_file_path}")

            # Upload image directories
            for image_dir in ["pages", "pictures", "tables"]:
                image_dir_path = local_output_dir / image_dir
                if image_dir_path.exists():
                    for image_file in image_dir_path.iterdir():
                        if image_file.is_file():
                            volume_image_path = f"{volume_output_path.rstrip('/')}/{image_dir}/{image_file.name}"

                            with open(image_file, "rb") as local_file:
                                self.workspace_client.files.upload(
                                    file_path=volume_image_path,
                                    contents=local_file,
                                    overwrite=True,
                                )

                            uploaded_files["images"].append(volume_image_path)
                            self.logger.info(f"Uploaded image: {volume_image_path}")

            return uploaded_files

        except Exception as e:
            self.logger.error(f"Failed to upload results: {str(e)}")
            raise

    def _cleanup_local_files(self, *paths: Path):
        """Clean up local temporary files."""
        for path in paths:
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        import shutil

                        shutil.rmtree(path)
                    self.logger.info(f"Cleaned up: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {path}: {str(e)}")


# MLflow model registration
if __name__ == "__main__":
    model = DoclingParsingModel()
    mlflow.pyfunc.log_model(
        artifact_path="docling_parser",
        python_model=model,
        conda_env={
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.12.3",
                "pip",
                {
                    "pip": [
                        "docling>=2.39.0",
                        "databricks-sdk",
                        "mlflow",
                        "pillow",
                    ]
                },
            ],
        },
    )
