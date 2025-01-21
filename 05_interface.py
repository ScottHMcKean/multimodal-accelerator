# Databricks notebook source
# MAGIC %md
# MAGIC # Interface
# MAGIC
# MAGIC This code sets up our Databricks app for running a chat interface and serving images of tables and pictures with a user response, as well as the retrieved document information.

# COMMAND ----------

import os
import yaml
import os
from src.maud.app import demo


def load_config(config_path="src/maud/app.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set environment variables from config
    for key, value in config.items():
        os.environ[key] = str(value)

    return config

if __name__ == "__main__":
    load_config()
    
    # Launch with specific configurations for local testing
    demo.launch(
        server_name="0.0.0.0",  # Allows external access
        server_port=7860,       # Default Gradio port
        share=False             # Set to True if you want a public URL
    )
