from pydantic import BaseModel
from typing import List
from pathlib import Path
import yaml


class InterfaceConfig(BaseModel):
    title: str
    description: str
    examples: List[str]


class AppConfig(BaseModel):
    serving_endpoint: str
    interface: InterfaceConfig


def load_config(config_path: str | Path) -> InterfaceConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed and validated configuration
    """
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
        return AppConfig.model_validate(raw_config)
