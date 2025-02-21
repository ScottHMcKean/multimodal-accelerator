# Databricks notebook source
# MAGIC %md
# MAGIC Installing dev requirements, initializing globals, adding maud path

# COMMAND ----------

!pip install -r requirements.txt --quiet
%restart_python

# COMMAND ----------

import sys
sys.path.append('./maud')

# COMMAND ----------

# UNITY CATALOG
CATALOG = 'shm'
SCHEMA = 'multimodal'

USERNAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

ROOT_PATH = "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split('/')[:-1])

# DOCLING SETTINGS
IMAGE_RESOLUTION_SCALE = 2.0
BRONZE_PATH = 'docs_bronze'
SILVER_PATH = 'docs_silver'
GOLD_PATH = 'docs_gold'

# CHUNK SETTINGS
CHUNK_MAX_TOKENS = 250

# COMMAND ----------

import logging

# Configure logger
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# COMMAND ----------

# Ensure volumes are ready
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

try:
    w.catalogs.create(name=CATALOG)
except:
    log.info(f"{CATALOG} catalog exists")

try:
    w.schemas.create(
    catalog_name=CATALOG,
    name=SCHEMA        
    )
except:
    log.info(f"{SCHEMA} catalog exists")

for volume_name in [BRONZE_PATH, SILVER_PATH, GOLD_PATH]:
    try:
        w.volumes.create(
        catalog_name=CATALOG, 
        schema_name=SCHEMA, 
        name=volume_name,
        volume_type=VolumeType.MANAGED
        )
    except:
        log.info(f"{volume_name} volume exists")
