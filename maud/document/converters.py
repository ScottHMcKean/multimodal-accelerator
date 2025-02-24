import hashlib
import json
from pathlib import Path
from typing import Dict
import logging

from pydantic import ConfigDict
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


class ExtendedDocument(DoclingDocument):
    page_metadata: Dict[int, MetaDataType] = {}
    input_hash: Path = None

    class Config:
        extra = "allow"


class PageMetadataModel:
    """
    A model that can be used to describe a page and extract hierarchy
    """

    def __init__(
        self,
        llm_client: OpenAI = None,
        llm_model: str = "gpt-4o-mini",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_pages(self, page_batch: Dict[int, PageItem]) -> Dict[int, MetaDataType]:
        if not self.enabled:
            return {}

        if not self.llm_client:
            self.logger.error("LLM client is required for page description")
            return {}

        page_metadata = {}
        for idx, page in page_batch.items():
            assert isinstance(page, PageItem)

            try:
                description = get_openai_description(
                    client=self.llm_client,
                    model=self.llm_model,
                    image=page.image.pil_image,
                    image_type="page",
                    max_tokens=200,
                )
            except:
                description = ""

            page_metadata[idx] = PictureDescriptionData(
                provenance=self.llm_model,
                text=description,
            )

        if not page_metadata:
            return {
                0: PictureDescriptionData(provenance="", text="No pages in document")
            }

        return page_metadata


class MaudConverter(DocumentConverter):
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        llm_client: OpenAI = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 200,
        overwrite: bool = False,
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
        self._hash_input()
        self._get_output_path()
        self.result = None

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

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
        if self._validate_output_exists():
            self.load_document()
            return self.document

        self.logger.info("Converting document")

        self.result = super().convert(self.input_path, *args, **kwargs)

        self.result.document = ExtendedDocument(
            **self.result.document.model_dump(),
            page_metadata=PageMetadataModel(
                llm_client=self.llm_client,
                llm_model=self.llm_model,
            ).analyze_pages(self.result.document.pages),
            input_hash=self._input_hash,
        )

        self.document = self.result.document
        return self.document

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

    def chunk(self):
        return chunk_maud_document(self.document, max_tokens=self.max_tokens)


class MAUDPipelineOptions(PdfPipelineOptions):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    images_scale: float = 2.0

    # pages
    generate_page_images: bool = True
    describe_pages: bool = True
    classify_pages: bool = True

    # pictures
    generate_picture_images: bool = True
    describe_pictures: bool = True
    classify_pictures: bool = True

    # tables
    generate_table_images: bool = True
    describe_tables: bool = True
    classify_tables: bool = True

    # llm
    llm_client: OpenAI = None
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 200

    # clf
    clf_client: OpenAI = None
    clf_model: str = "yolo_v8"


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
    def __init__(self, pipeline_options: MAUDPipelineOptions):
        self.enabled = pipeline_options.describe_pictures
        self.llm_client = pipeline_options.llm_client
        self.llm_model = pipeline_options.llm_model
        self.max_tokens = pipeline_options.max_tokens

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem) and self.llm_client

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            assert isinstance(element, PictureItem)

            try:
                description = get_openai_description(
                    client=self.llm_client,
                    model=self.llm_model,
                    image=element.image.pil_image,
                    image_type="picture",
                    max_tokens=200,
                )
            except:
                description = ""

            element.annotations.append(
                PictureDescriptionData(
                    provenance=self.llm_model,
                    text=description,
                )
            )

            yield element


class PictureClassifierModel(BaseEnrichmentModel):
    def __init__(self, pipeline_options: MAUDPipelineOptions):
        self.enabled = pipeline_options.classify_pictures
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


class TableDescriptionModel(BaseEnrichmentModel):
    def __init__(self, pipeline_options: MAUDPipelineOptions):
        self.enabled = pipeline_options.describe_tables
        self.llm_client = pipeline_options.llm_client
        self.llm_model = pipeline_options.llm_model
        self.max_tokens = pipeline_options.max_tokens

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, TableItem) and self.llm_client

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            assert isinstance(element, TableItem)

            try:
                description = get_openai_description(
                    client=self.llm_client,
                    model=self.llm_model,
                    image=element.image.pil_image,
                    image_type="table",
                    max_tokens=200,
                )
            except:
                description = ""

            caption = doc.add_text(
                label=DocItemLabel.CAPTION,
                text=description,
                orig=description,
                prov=element.prov[0],
            )

            element.captions.append(caption.get_ref())

            yield element


class MAUDPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: MAUDPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: pipeline_options

        self.enrichment_pipe = [
            PictureClassifierModel(self.pipeline_options),
            PictureDescriptionModel(self.pipeline_options),
            TableDescriptionModel(self.pipeline_options),
        ]

    @classmethod
    def get_default_options(cls) -> MAUDPipelineOptions:
        return MAUDPipelineOptions()
