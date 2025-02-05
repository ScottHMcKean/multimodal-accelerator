"""Configuration management for MAUD."""
from pathlib import Path
import yaml
from typing import Dict, Any
import mlflow

def load_config(config_type, config_file='config.yaml'):
  """
  A helper function to access different configurations
  from the mlflow config file.
  """
  config = mlflow.models.ModelConfig(development_config=config_file)

  config_mapping = {"model": config.get("model"),
                    "langgraph": config.get("langgraph"),
                    "retriever": config.get("retriever"),
                    "mlflow": config.get("mlflow")}
  
  config_types = config_mapping.keys()
  
  if config_type in config_types:    
    return config_mapping.get(config_type)
  
  else:
    raise Exception(
      f"The config type, {config_type}, is not among the supported types, ({(", ").join(config_types)})."
    )

CONFIG_DIR = Path(__file__).parent

class Config:
    """Configuration container with nested attribute access."""
    
    def __init__(self, config_name: str):
        """
        Initialize configuration from a YAML file.
        
        Args:
            config_name: Name of the config file (e.g., "app_config.yaml")
        """
        self._config = self._load_config(config_name)
        self._convert_to_attributes(self._config)
    
    def _load_config(self, config_name: str) -> Dict[Any, Any]:
        """
        Load and parse a YAML configuration file.
        
        Args:
            config_name: Name of the config file (e.g., "app_config.yaml")
        """
        config_path = CONFIG_DIR / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            return config if config is not None else {}
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file {config_path}: {str(e)}")
    
    def _convert_to_attributes(self, config_dict: Dict[str, Any]) -> None:
        """
        Recursively convert dictionary to nested attributes.
        
        Args:
            config_dict: Dictionary to convert to attributes
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Create nested Config object for dictionaries
                nested_config = Config.__new__(Config)  # Create without calling __init__
                nested_config._config = value
                nested_config._convert_to_attributes(value)
                setattr(self, key, nested_config)
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with optional default.
        
        Args:
            key: Key to get the value from
            default: Default value to return if key is not found
        """
        return getattr(self, key, default)
    
    def __getattr__(self, name: str) -> Any:
        """
        Handle attribute access for missing attributes.
        
        Args:
            name: Name of the attribute to get
        """
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __repr__(self) -> str:
        """
        String representation of the config.
        
        Returns:
            str: String representation of the config
        """
        return f"Config({self._config})" 