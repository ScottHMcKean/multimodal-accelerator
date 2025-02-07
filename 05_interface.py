# Databricks notebook source
# MAGIC %md
# MAGIC # Interface
# MAGIC
# MAGIC This code sets up our Databricks app for running a chat interface and serving images of tables and pictures with a user response, as well as the retrieved document information.
# MAGIC
# MAGIC There two key concepts here:
# MAGIC - How Databricks Apps work
# MAGIC - How to work with a multimodal chat interface

# COMMAND ----------

# MAGIC %md
# MAGIC ## How Databricks Apps Work
# MAGIC
# MAGIC Databricks apps are a hosted container service. They use a yaml configuration file (`maud\interface\app.yaml`) to specify a run path where we run a main python file. This pattern is universal for pretty much all front ends written in python. In this accelerator we leverage Gradio, specifically the `ChatInterface` object with multimodal mode to share the retrieved images directly in chat. This is highly customizable using general UI frameworks like Streamlit, Gradio, or Flask.
# MAGIC
# MAGIC The code below uses the Databricks CLI to create and deploy the app. This pattern is useful because it can be replicated in continous integration / continous deployment (CI/CD) systems.

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from maud.interface.create import create_app

w = WorkspaceClient()

# COMMAND ----------

app_name = "multimodal_maud"
source_code_path = "maud/interfaces"
try:
    app_info = w.apps.get(app_name)
except Exception as e:
    print(e)
    create_app()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to work with a multimodal chat interface
# MAGIC
# MAGIC We use our Databricks App as a front end for a chat interface. This a nice abstraction for connecting to an agent, like the one we designed using LangGraph in inference and in `maud/agent`. At the core of this abstraction is an API call to the serving endpoint that is hosting the agent.
# MAGIC
# MAGIC The pattern for this works like so:
# MAGIC
# MAGIC UI --> [Message] --> API --> [Message] --> UI
# MAGIC
# MAGIC But when dealing with detailed document retrieval, we need to ensure that we are passing images of the pages, tables, and pictures back to the UI, along with the LLM summary based on the augmented context.

# COMMAND ----------
