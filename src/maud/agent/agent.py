from langchain.prompts import PromptTemplate
from open import ChatDatabricks
import mlflow
from maud.config import load_config, Config

# Load configuration
config = Config(load_config("agent_config.yaml"))

# Initialize Databricks LLM
databricks_llm = ChatDatabricks(
    endpoint=config.llm.endpoint,
    temperature=config.llm.temperature
)

# Create the prompt and chain using RunnableSequence
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=config.prompts.template_instruction,
)
lc_agent = prompt | databricks_llm

# Set the model in MLflow
mlflow.models.set_model(lc_agent)