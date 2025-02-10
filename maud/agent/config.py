from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import mlflow
from pathlib import Path


class ConfigModel(BaseModel):
    """
    We use pydantic to help standardize config files.
    We use the ConfigDict to allow extra fields in the config files.
    This also provides extensibility.
    https://docs.pydantic.dev/latest/api/config/
    """

    model_config = ConfigDict(extra="allow")


class ModelParameters(ConfigModel):
    temperature: float
    max_tokens: int


class ModelConfig(ConfigModel):
    endpoint_name: str
    parameters: ModelParameters


class RetrieverMapping(ConfigModel):
    chunk_text: str
    document_uri: str
    primary_key: str
    other_columns: List[str]

    @property
    def all_columns(self) -> List[str]:
        """
        Combines chunk_text, document_uri, primary_key and other_columns
        into a single list of all columns.
        """
        return [
            self.chunk_text,
            self.document_uri,
            self.primary_key,
        ] + self.other_columns


class RetrieverParameters(ConfigModel):
    k: int = 5
    query_type: str = "ann"
    score_threshold: float = 0


class RetrieverConfig(ConfigModel):
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None
    endpoint_name: str
    index_name: str
    embedding_model: str
    parameters: RetrieverParameters
    mapping: RetrieverMapping
    chunk_template: str = "Passage: {chunk_text}\n Document URI: {document_uri}\n"


class AgentConfig(ConfigModel):
    streaming: bool = False
    experiment_location: Optional[str] = None
    uc_model_name: Optional[str] = None


class MaudConfig(ConfigModel):
    agent: AgentConfig
    model: ModelConfig
    retriever: RetrieverConfig


def parse_config(mlflow_config: mlflow.models.ModelConfig) -> MaudConfig:
    """
    Parse an mlflow config into a pydantic configuration.
    """
    return MaudConfig(**mlflow_config.to_dict())
