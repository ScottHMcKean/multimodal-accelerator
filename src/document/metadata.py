from typing import Literal, Annotated, Union, Optional

from pydantic import BaseModel, Field
from docling_core.types.doc import PictureClassificationData
from docling_core.types.doc.document import PictureDescriptionData


class RelationshipData(BaseModel):
    """Common relationship types in knowledge graphs."""

    kind: Literal["relationship"] = "relationship"
    IS_A: str = "is_a"
    PART_OF: str = "part_of"
    HAS_PROPERTY: str = "has_property"
    RELATED_TO: str = "related_to"
    CAUSES: str = "causes"
    DEPENDS_ON: str = "depends_on"


class NodeData(BaseModel):
    kind: Literal["entity"] = "entity"
    id: str
    label: str
    type: str
    properties: Optional[dict]


class ConnectionData(BaseModel):
    """Connection between two nodes for a knowledge graph."""

    kind: Literal["connection"] = "connection"
    subject: NodeData
    predicate: RelationshipData
    object: NodeData


class PageMetadataData(BaseModel):
    """Page metadata from LLM analysis."""

    kind: Literal["page_metadata"] = "page_metadata"
    description: str
    processed_at: float
    model: Optional[str] = None
    error: Optional[str] = None


MetaDataType = Annotated[
    Union[
        PictureClassificationData,
        PictureDescriptionData,
        NodeData,
        RelationshipData,
        ConnectionData,
        PageMetadataData,
    ],
    Field(discriminator="kind"),
]
