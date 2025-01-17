from abc import ABC, abstractmethod
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DoclingDocument
from docling_core.types.doc import ImageRefMode

class AbstractConverter(ABC):
    def __init__(self, input_path: Path, output_dir: Path, overwrite=False):
        self.doc_file_name = 'doc.md'
        self.input_path = input_path
        self.output_dir = output_dir
        self.overwrite = overwrite
        self._hash_input()
        self._get_output_path()
        self.result = None

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

    @abstractmethod
    def convert(self, overwrite=False):
        pass

    @abstractmethod
    def save_result(self):
        pass

    @abstractmethod
    def load_result(self):
        pass

class DoclingConverterAdapter(AbstractConverter):
    def __init__(self, input_path: Path, output_dir: Path, overwrite=False, **kwargs):
        super().__init__(input_path, output_dir, overwrite)
        self.converter = DocumentConverter(**kwargs)
        self.doc_file_name = 'doc.json'
        self.md_file_name = 'doc.md'
    
    def convert(self):
        if self._validate_output_exists():
            self.result = self.load_result()
            return self.result
        
        print('converting document')
        self.result = self.converter.convert(self.input_path)
        self.document = self.result.document
        return self.document

    def load_result(self):
        print('loading document')
        
        with (self._output_path / self.doc_file_name).open("r") as fp:
            doc_dict = json.loads(fp.read())
        
        self.document = DoclingDocument.model_validate(doc_dict)
        return self.document
      
    def save_result(self):
        print('saving document')

        self._output_path.mkdir(parents=False, exist_ok=True)

        self.document.save_as_markdown(
            self._output_path / self.md_file_name,
            image_mode=ImageRefMode.EMBEDDED
            )
    
        # json_path = self._output_path / self.doc_file_name
        self.document.save_as_json(
            self._output_path / self.doc_file_name, 
            image_mode=ImageRefMode.EMBEDDED
            )

        return True