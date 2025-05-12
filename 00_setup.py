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

CATALOG = 'shm'
SCHEMA = 'osc'
RAW_DOCS_VOL = 'raw_docs'
PROCESSED_DOCS_VOL = 'processed_docs'

# COMMAND ----------

USERNAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

ROOT_PATH = "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split('/')[:-1])

# COMMAND ----------

import logging

# Configure logger
logging.basicConfig(
  level=logging.WARNING, 
  format='%(asctime)s - %(levelname)s - %(message)s'
  )
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

for vol_name in [RAW_DOCS_VOL, PROCESSED_DOCS_VOL]:
    try:
        w.volumes.create(
        catalog_name=CATALOG, 
        schema_name=SCHEMA, 
        name=vol_name,
        volume_type=VolumeType.MANAGED
        )
    except:
        log.info(f"{vol_name} volume exists")
