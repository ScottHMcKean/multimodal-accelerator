# Databricks notebook source
# MAGIC %md
# MAGIC Installing dev requirements, initializing globals, adding dbmma path

# COMMAND ----------

!pip install -r requirements-dev.txt --quiet
%restart_python

# COMMAND ----------

import sys
sys.path.append('./src')

# COMMAND ----------

# UNITY CATALOG
CATALOG = 'shm'
SCHEMA = 'multimodal'

# USER INFO
USERNAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

SETUP_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

REPO_NAME = SETUP_PATH.split("/")[-2]

# DOCLING SETTINGS
IMAGE_RESOLUTION_SCALE = 2.0
BRONZE_PATH = 'docs_bronze'
SILVER_PATH = 'docs_silver'
GOLD_PATH = 'docs_gold'

# COMMAND ----------


