from pydantic import BaseModel
from typing import List
from pathlib import Path
import yaml


class InterfaceConfig(BaseModel):
    title: str
    description: str
    example: str
    serving_endpoint: str


def load_config(config_path: str | Path) -> InterfaceConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed and validated configuration
    """
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
        return InterfaceConfig.model_validate(raw_config)
