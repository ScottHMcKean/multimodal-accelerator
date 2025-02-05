"""Extended document types for MAUD."""
from docling_core.types.doc import Size
from docling_core.types.doc.document import PageItem, ImageRef
from typing import Optional, Dict, Any

class ExtendedPageItem(PageItem):
    """Extended PageItem with additional metadata."""
    
    def __init__(
        self,
        page_no: int,
        size: Size,
        image: Optional[ImageRef] = None,
        metadata: Optional[Dict[str, Any]] = None
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

class ExtendedDocument:
    """Wrapper for DoclingDocument with extended page functionality."""
    
    def __init__(self, doc: DoclingDocument):
        self.doc = doc
        self.page_metadata: Dict[int, PageMetadata] = {}
        
    def add_page_metadata(self, page_no: int, key: str, value: Any) -> None:
        """Add metadata to a page."""
        if page_no not in self.page_metadata:
            if page_no not in self.doc.pages:
                raise KeyError(f"Page {page_no} not found in document")
            self.page_metadata[page_no] = PageMetadata(self.doc.pages[page_no])
        self.page_metadata[page_no].add_metadata(key, value)
        
    def get_page_metadata(self, page_no: int, key: str) -> Any:
        """Get metadata from a page."""
        if page_no not in self.page_metadata:
            return None
        return self.page_metadata[page_no].get_metadata(key) 