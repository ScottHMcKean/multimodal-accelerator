import hashlib
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DoclingDocument
from docling_core.types.doc import ImageRefMode, PageItem
import logging
from docling.datamodel.pipeline_options import PdfPipelineOptions
from maud.document.metadata import MetaDataType, DescriptionData
from typing import Dict


class ExtendedDocument(DoclingDocument):
    page_metadata: Dict[int, MetaDataType] = {}

    class Config:
        extra = "allow"


class PageMetadataModel:
    """
    A model that can be used to describe a page and extract hierarchy
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def analyze_pages(self, page_batch: Dict[int, PageItem]) -> Dict[int, MetaDataType]:
        if not self.enabled:
            return {}

        page_metadata = {}
        for idx, page in page_batch.items():
            assert isinstance(page, PageItem)
            page_metadata[idx] = DescriptionData(
                provenance="a llm model",
                text="This is a dummy description",
            )

        if not page_metadata:
            return {0: DescriptionData(provenance="", text="No pages in document")}

        return page_metadata


class MaudConverter(DocumentConverter):
    def __init__(self, input_path: Path, output_dir: Path, overwrite=False, **kwargs):
        super().__init__(**kwargs)
        self.doc_file_name = "doc.json"
        self.md_file_name = "doc.md"
        self.input_path = input_path
        self.output_dir = output_dir
        self.overwrite = overwrite
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

        return True

    def convert(self, *args, **kwargs):
        self.logger.info("Converting document")

        if self._validate_output_exists():
            self.logger.info("Conversion exists, reloading document")
            self.document = self.load_document()

        self.result = super().convert(self.input_path, *args, **kwargs)

        self.result.document = ExtendedDocument(
            **self.result.document.model_dump(),
            page_metadata=PageMetadataModel().analyze_pages(self.result.document.pages),
        )

        self.document = self.result.document

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


class MAUDPipelineOptions(PdfPipelineOptions):

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
