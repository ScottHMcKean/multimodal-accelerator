import mlflow
from databricks_langchain import ChatDatabricks
from config_utils import load_config


config = load_config("model")
model_name = config.get('name')
model_config = config.get('parameters')

model = ChatDatabricks(endpoint=model_name, extra_params=model_config)