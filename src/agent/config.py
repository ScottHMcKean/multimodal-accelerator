from pydantic import BaseModel, ConfigDict, Field
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


class DataConfig(ConfigModel):
    uc_catalog: str = Field(alias="catalog")
    uc_schema: str = Field(alias="schema")
    raw_docs_vol: str
    processed_docs_vol: str
    chunks_table_name: str
    overwrite: bool = False


class ModelConfig(ConfigModel):
    endpoint_name: str
    temperature: float
    max_tokens: int


class RetrieverConfig(ConfigModel):
    endpoint_name: str
    index_name: str
    embedding_model: str
    search_type: str
    score_threshold: float
    num_results: int
    text_column: str
    document_uri: str
    primary_key: str
    chunk_template: str
    additional_columns: List[str]

    @property
    def all_columns(self) -> List[str]:
        """
        Combines chunk_text, document_uri, primary_key and other_columns
        into a single list of all columns.
        """
        return [
            self.text_column,
            self.document_uri,
            self.primary_key,
        ] + self.additional_columns


class AgentConfig(ConfigModel):
    streaming: bool = False
    experiment_location: Optional[str] = None
    uc_model_name: Optional[str] = None


class InterfaceConfig(ConfigModel):
    title: str
    description: str
    example: str
    serving_endpoint: str


class MaudConfig(ConfigModel):
    data: DataConfig
    model: ModelConfig
    retriever: RetrieverConfig
    agent: AgentConfig
    interface: InterfaceConfig


def parse_config(mlflow_config: mlflow.models.ModelConfig) -> MaudConfig:
    """
    Parse an mlflow config into a pydantic configuration.
    """
    return MaudConfig(**mlflow_config.to_dict())
