import hashlib
import json
from pathlib import Path
from typing import Dict
import logging
import os
import time
import gc
from threading import Lock
import itertools

from pydantic import ConfigDict, model_validator
from openai import OpenAI
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DoclingDocument
from docling_core.types.doc import ImageRefMode, PageItem
from docling_core.types.doc.document import PictureDescriptionData
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import Size
from docling_core.types.doc.document import PageItem, ImageRef, DoclingDocument
from typing import Optional, Dict, Any
from typing import Any, Iterable

from docling_core.types.doc import DoclingDocument, NodeItem, PictureItem, TableItem
from docling_core.types.doc.document import (
    PictureDescriptionData,
    PictureClassificationData,
    PictureClassificationClass,
)
from docling.models.base_model import BaseEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc.document import DocItemLabel

from maud.document.extensions import get_openai_description
from maud.document.metadata import MetaDataType
from maud.document.chunkers import chunk_maud_document

# Module-level cache for model instances
_MODEL_CACHE = {}
_CACHE_LOCK = Lock()


def get_cached_model(cache_key: str, factory_func):
    """Thread-safe model caching to avoid repeated initialization."""
    with _CACHE_LOCK:
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = factory_func()
        return _MODEL_CACHE[cache_key]


class ExtendedDocument(DoclingDocument):
    page_metadata: Dict[int, MetaDataType] = {}
    input_hash: Path = None

    model_config = ConfigDict(extra="allow")


class PageMetadataModel:
    """
    Simple model for page analysis with basic batch processing.
    """

    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        llm_model: str = "gpt-4o-mini",
        enabled: bool = True,
        batch_size: int = 5,
        max_retries: int = 3,
    ):
        self.enabled = enabled
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_pages(
        self, page_batch: Dict[int, PageItem]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze pages in simple batches.
        """
        if not self.enabled or not self.llm_client:
            return {}

        results = {}
        page_items = list(page_batch.items())

        # Process in batches
        for i in range(0, len(page_items), self.batch_size):
            batch = page_items[i : i + self.batch_size]

            for page_idx, page in batch:
                try:
                    if page.image is not None:
                        description = self._get_description_with_retry(
                            page.image, page_idx
                        )
                        results[page_idx] = {
                            "description": description,
                            "processed_at": time.time(),
                            "model": self.llm_model,
                        }
                    else:
                        results[page_idx] = {
                            "description": "No image available",
                            "processed_at": time.time(),
                        }
                except Exception as e:
                    self.logger.error(f"Failed to process page {page_idx}: {e}")
                    results[page_idx] = {
                        "description": "Processing failed",
                        "error": str(e),
                        "processed_at": time.time(),
                    }

        return results

    def _get_description_with_retry(self, image, page_idx: int) -> str:
        """
        Get page description with simple retry logic.
        """
        if not image:
            return "No image available"

        prompt = (
            "Analyze this document page and provide a concise summary of:\n"
            "1. Document type and purpose\n"
            "2. Key content elements\n"
            "3. Notable visual features\n"
            "Be precise and factual."
        )

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    time.sleep(2**attempt)  # Simple exponential backoff

                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image.doclaynet_url},
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                    temperature=0.1,
                    timeout=30,
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for page {page_idx}: {str(e)}"
                )
                if attempt == self.max_retries - 1:
                    return f"Failed to generate description after {self.max_retries} attempts"

        return "Description unavailable"


class MAUDConverter(DocumentConverter):
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        llm_client: OpenAI = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 200,
        overwrite: bool = False,
        enable_caching: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.doc_file_name = "doc.json"
        self.md_file_name = "doc.md"
        self.input_path = input_path
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.enable_caching = enable_caching
        self._hash_input()
        self._get_output_path()
        self.result = None

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.saved_file_locations = {"pages": {}, "pictures": {}, "tables": {}}

    def _hash_input(self):
        self._input_hash = self._generate_input_hash(self.input_path)

    def _get_output_path(self):
        self._output_path = self.output_dir / self._input_hash

    def _generate_input_hash(self, input_path: Path):
        return hashlib.md5(str(input_path).encode()).hexdigest()

    def _validate_output_exists(self):
        if self.overwrite:
            return False

        if not self._output_path.exists():
            return False

        if not (self._output_path / self.doc_file_name).exists():
            return False

        self.logger.info("Found existing conversion")

        return True

    def convert(self, *args, **kwargs):
        """Convert with basic optimizations."""
        if self._validate_output_exists():
            self.load_document()
            return self.document

        self.logger.info(f"Converting document: {self.input_path}")

        try:
            self.result = super().convert(self.input_path, *args, **kwargs)

            # Check if page description is enabled
            try:
                enabled = self.format_to_options[
                    "pdf"
                ].pipeline_options.do_page_description
            except (KeyError, AttributeError):
                enabled = False

            # Use page metadata model if enabled
            if enabled and self.llm_client:
                page_metadata_model = PageMetadataModel(
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    enabled=enabled,
                    batch_size=5,
                    max_retries=3,
                )
                page_metadata = page_metadata_model.analyze_pages(
                    self.result.document.pages
                )
            else:
                page_metadata = {}

            self.result.document = ExtendedDocument(
                **self.result.document.model_dump(),
                page_metadata=page_metadata,
                input_hash=self._input_hash,
            )

            self.document = self.result.document
            self.logger.info(f"Document conversion completed")

            return self.document

        except Exception as e:
            self.logger.error(f"Document conversion failed: {e}")
            raise
        finally:
            # Cleanup to prevent memory leaks
            gc.collect()

    def load_document(self):
        self.logger.info("Loading document")

        with (self._output_path / self.doc_file_name).open("r") as fp:
            doc_dict = json.loads(fp.read())

        self.document = ExtendedDocument.model_validate(doc_dict)

    def save_document(self):
        self.logger.info("Saving document")

        self._output_path.mkdir(parents=True, exist_ok=True)

        self.document.save_as_markdown(
            self._output_path / self.md_file_name, image_mode=ImageRefMode.EMBEDDED
        )

        self.document.save_as_json(
            self._output_path / self.doc_file_name, image_mode=ImageRefMode.EMBEDDED
        )

        self.save_images()

    def save_images(self):
        """Save images with basic error handling."""
        directories = {
            "pages": self._output_path / "pages",
            "pictures": self._output_path / "pictures",
            "tables": self._output_path / "tables",
        }

        for dir_path in directories.values():
            dir_path.mkdir(exist_ok=True, parents=True)

        # Save page images
        for page_no, page in self.document.pages.items():
            if page.image is not None:
                try:
                    page_image_path = directories["pages"] / f"{page_no}.webp"
                    with page_image_path.open("wb") as fp:
                        page.image.pil_image.save(fp, format="webp", quality=85)
                    self.saved_file_locations["pages"][page_no] = str(page_image_path)
                except Exception as e:
                    self.logger.error(f"Failed to save page {page_no} image: {e}")

        # Save picture images
        for picture in self.document.pictures:
            if picture.image is not None:
                try:
                    pic_ref = picture.self_ref.split("/")[-1]
                    pic_image_path = directories["pictures"] / f"{pic_ref}.webp"
                    with pic_image_path.open("wb") as fp:
                        picture.image.pil_image.save(fp, format="webp", quality=85)
                    self.saved_file_locations["pictures"][pic_ref] = str(pic_image_path)
                except Exception as e:
                    self.logger.error(f"Failed to save picture {pic_ref} image: {e}")

        # Save table images
        for table in self.document.tables:
            if table.image is not None:
                try:
                    table_ref = table.self_ref.split("/")[-1]
                    table_image_path = directories["tables"] / f"{table_ref}.webp"
                    with table_image_path.open("wb") as fp:
                        table.image.pil_image.save(fp, format="webp", quality=85)
                    self.saved_file_locations["tables"][table_ref] = str(
                        table_image_path
                    )
                except Exception as e:
                    self.logger.error(f"Failed to save table {table_ref} image: {e}")

    def chunk(self):
        return chunk_maud_document(
            self.document,
            max_tokens=self.max_tokens,
            image_locations=self.saved_file_locations,
        )


class MAUDPipelineOptions(PdfPipelineOptions):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    images_scale: float = 2.0
    do_page_description: bool = False

    # llm
    llm_client: Optional[OpenAI] = None
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 200

    # clf
    clf_client: Optional[OpenAI] = None
    clf_model: str = "yolo_v8"

    @model_validator(mode="after")
    def validate_picture_description(self):
        if self.do_picture_description and not self.generate_picture_images:
            raise ValueError(
                "do_picture_description requires generate_picture_images to be enabled"
            )
        return self

    @model_validator(mode="after")
    def validate_page_description(self):
        if self.do_page_description and not self.generate_page_images:
            raise ValueError(
                "do_page_description requires generate_page_images to be enabled"
            )
        return self


class ExtendedPageItem(PageItem):
    """Extended PageItem with additional metadata."""

    def __init__(
        self,
        page_no: int,
        size: Size,
        image: Optional[ImageRef] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(page_no=page_no, size=size, image=image)
        self.metadata = metadata or {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the page."""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get metadata from the page."""
        return self.metadata.get(key)


class PageMetadata:
    """Companion class to store additional page metadata."""

    def __init__(self, page: PageItem):
        self.page = page
        self.metadata: Dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the page."""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get metadata from the page."""
        return self.metadata.get(key)


class PictureDescriptionModel(BaseEnrichmentModel):
    """
    Simple picture description model with basic retry logic.
    """

    def __init__(self, pipeline_options):
        self.enabled = pipeline_options.do_picture_description
        self.llm_client = pipeline_options.llm_client
        self.llm_model = pipeline_options.llm_model
        self.max_tokens = pipeline_options.max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        is_picture = isinstance(element, PictureItem)
        has_client = bool(self.llm_client)
        has_image = is_picture and element.image is not None
        return self.enabled and is_picture and has_client and has_image

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            if self.is_processable(doc, element):
                assert isinstance(element, PictureItem)

                description = self._get_description_with_retry(element)
                element.text = description

                yield element

    def _get_description_with_retry(self, element: PictureItem) -> str:
        """Generate description with simple retry logic."""
        if not element.image:
            return "No image available for description"

        prompt = (
            "Describe this image concisely in 2-3 sentences. Focus on:\n"
            "- Main subjects/objects\n"
            "- Key visual elements\n"
            "- Context or setting\n"
            "Be factual and specific."
        )

        max_retries = 3

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2**attempt)  # Simple exponential backoff

                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": element.image.doclaynet_url},
                                },
                            ],
                        }
                    ],
                    max_tokens=min(self.max_tokens, 200),
                    temperature=0.1,
                    timeout=45,
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                self.logger.warning(
                    f"Picture description attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == max_retries - 1:
                    return f"Failed to generate description"

        return "Description unavailable"


class PictureClassifierModel(BaseEnrichmentModel):
    def __init__(self, pipeline_options: MAUDPipelineOptions):
        self.enabled = pipeline_options.do_picture_classification
        self.clf_client = pipeline_options.clf_client
        self.clf_model = pipeline_options.clf_model

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem) and self.clf_client

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            assert isinstance(element, PictureItem)

            element.annotations.append(
                PictureClassificationData(
                    provenance="example_classifier-0.0.1",
                    predicted_classes=[
                        PictureClassificationClass(class_name="dummy", confidence=0.42)
                    ],
                )
            )

            yield element


class MAUDPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: MAUDPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options = pipeline_options

        self.enrichment_pipe = [
            PictureClassifierModel(self.pipeline_options),
            PictureDescriptionModel(self.pipeline_options),
        ]

    @classmethod
    def get_default_options(cls) -> MAUDPipelineOptions:
        return MAUDPipelineOptions()
