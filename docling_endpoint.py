from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from databricks.sdk import WorkspaceClient
from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode
from mlflow.pyfunc import PythonModel


class DoclingParsingModel(PythonModel):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_client: WorkspaceClient | None = None
        self.temp_dir: Path | None = None

    def load_context(self, context) -> None:
        self.logger.info("Loading Docling parsing model context")
        self.workspace_client = WorkspaceClient()
        self.temp_dir = Path(tempfile.mkdtemp())

    def predict(self, context, model_input) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        inputs = self._normalize_inputs(model_input)

        for input_data in inputs:
            try:
                file_path = input_data.get("file_path")
                output_path = input_data.get("output_path")
                output_root = input_data.get("output_root") or os.getenv(
                    "DOCLING_OUTPUT_ROOT"
                )

                if not file_path:
                    raise ValueError("file_path is required")

                if not output_path:
                    if output_root:
                        output_path = (
                            f"{output_root.rstrip('/')}/"
                            f"{self._sanitize_filename(Path(file_path).stem)}"
                        )
                    else:
                        raise ValueError("output_path or output_root is required")

                local_file_path = self._download_file_from_volume(file_path)
                local_output_dir = (
                    self.temp_dir
                    / "output"
                    / self._sanitize_filename(Path(file_path).stem)
                )
                local_output_dir.mkdir(parents=True, exist_ok=True)

                converter = DocumentConverter()
                result = converter.convert(source=local_file_path)
                document = result.document

                self._save_document_locally(document, local_output_dir)
                upload_results = self._upload_results_to_volume(
                    local_output_dir=local_output_dir,
                    volume_output_path=output_path,
                )

                self._cleanup_local_files(local_file_path, local_output_dir)

                results.append(
                    {
                        "status": "success",
                        "input_path": file_path,
                        "output_path": output_path,
                        "uploaded_files": upload_results,
                    }
                )
            except Exception as exc:
                self.logger.error("Error processing document: %s", exc)
                results.append(
                    {
                        "status": "error",
                        "error": str(exc),
                        "input_path": input_data.get("file_path", "unknown"),
                    }
                )

        return results

    def _normalize_inputs(self, model_input: Any) -> List[Dict[str, Any]]:
        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")
        if isinstance(model_input, dict):
            return [model_input]
        if isinstance(model_input, list):
            if not model_input:
                return []
            if isinstance(model_input[0], dict):
                return model_input
            if isinstance(model_input[0], str):
                return [{"file_path": item} for item in model_input]
        raise ValueError("Unsupported model input format")

    def _download_file_from_volume(self, volume_path: str) -> Path:
        if not self.workspace_client or not self.temp_dir:
            raise RuntimeError("Model context not loaded")

        local_path = self.temp_dir / "input" / Path(volume_path).name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with self.workspace_client.files.download(volume_path) as download_response:
            with open(local_path, "wb") as local_file:
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:
                        local_file.write(chunk)
        return local_path

    def _save_document_locally(self, document, output_dir: Path) -> None:
        json_file = output_dir / "doc.json"
        document.save_as_json(json_file, image_mode=ImageRefMode.EMBEDDED)

        md_file = output_dir / "doc.md"
        document.save_as_markdown(md_file, image_mode=ImageRefMode.EMBEDDED)

    def _upload_results_to_volume(
        self, local_output_dir: Path, volume_output_path: str
    ) -> Dict[str, List[str]]:
        if not self.workspace_client:
            raise RuntimeError("Model context not loaded")

        uploaded_files = {"json": [], "markdown": []}
        for file_path in local_output_dir.iterdir():
            if file_path.is_file():
                volume_file_path = f"{volume_output_path.rstrip('/')}/{file_path.name}"
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

        return uploaded_files

    def _cleanup_local_files(self, *paths: Path) -> None:
        for path in paths:
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        import shutil

                        shutil.rmtree(path)
            except Exception as exc:
                self.logger.warning("Failed to cleanup %s: %s", path, exc)

    def _sanitize_filename(self, filename: str) -> str:
        stem = Path(filename).stem.replace("%20", "_")
        sanitized = re.sub(r"[^\w\s-]", "", stem)
        sanitized = sanitized.replace(" ", "_")
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_").lower()
        return sanitized
