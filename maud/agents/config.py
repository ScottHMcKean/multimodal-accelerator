from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any


class ConfigModel(BaseModel):
    """
    We use pydantic to help standardize config files.
    We use the ConfigDict to allow extra fields in the config files.
    This also provides extensibility.
    https://docs.pydantic.dev/latest/api/config/
    """
    model_config = ConfigDict(extra='allow')

class ModelParameters(ConfigModel):
    max_tokens: int
    temperature: float

class ModelConfig(ConfigModel):
    name: str
    parameters: ModelParameters

class RetrieverSchema(ConfigModel):
    chunk_text: str
    document_uri: str
    primary_key: str

class RetrieverParameters(ConfigModel):
    k: int
    query_type: str

class RetrieverConfig(ConfigModel):
    vector_search_endpoint_name: str
    vector_search_index: str
    embedding_model: str
    parameters: RetrieverParameters
    schema: RetrieverSchema
    chunk_template: str

class MLFlowConfig(ConfigModel):
    experiment_location: str
    uc_model: str

class LanggraphConfig(ConfigModel):
    model: ModelConfig
    langgraph: Dict[str, Any]
    retriever: RetrieverConfig
    mlflow: MLFlowConfig

class LLMConfig(ConfigModel):
    endpoint_name: str
    temperature: float
    max_tokens: int

class VectorSearchConfig(ConfigModel):
    endpoint_name: str
    combined_chunks_table: str
    index_name: str
    num_results: int
    score_threshold: float
    search_type: str
    query_type: str
    primary_key: str
    text_column: str
    doc_uri: str
    other_columns: List[str]

class AgentConfig(ConfigModel):
    llm: LLMConfig
    vector_search: VectorSearchConfig
    system_prompt: str