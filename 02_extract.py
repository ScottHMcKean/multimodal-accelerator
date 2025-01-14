# Databricks notebook source
# MAGIC %md
# MAGIC # Extract
# MAGIC
# MAGIC This module processes our bronze documents. It is the most involved of the modules but we leverage Docling as a framework to abstract away the layout analysis of a document. Docling takes a list of files in the volumes, parallelized over workers.
# MAGIC
# MAGIC In order to make this result useful downstream, we need to do three things:
# MAGIC
# MAGIC - Export the result to a reloadable json format
# MAGIC - Save and export the images
# MAGIC - Save and export the tables
# MAGIC
# MAGIC In order to get the exports, we can modify the defaults and generate page, picture, and table images. This really slows down the parsing pipeline, but is essential for our user interface to reload and serve everything.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC This section runs through the parsing and export of a single document. This takes a while since we are OCRing, extracting images, and analyzing the layout of each document. Docling provides a nice framework for exporting these documents as markdown files, both with linked and embedded images. This makes the downstream take of summarizing and converting images to text much easier, where we can even replace the images with a list of symbols references, description, caption etc.

# COMMAND ----------

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import DoclingDocument

# setup the conversion pipeline to extract images and tables automatically
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = False
pipeline_options.generate_table_images = False

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# COMMAND ----------

from pathlib import Path
test_path = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}/FAA-2021-0268-0002.pdf")
result = doc_converter.convert(test_path)
output_dir = Path(f'/Volumes/{CATALOG}/{SCHEMA}/{SILVER_PATH}')
doc_name = test_path.stem

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's analyze what we are getting for results. We can break this into metadata and parsed. 
# MAGIC
# MAGIC Metadata includes:
# MAGIC - Schema_name
# MAGIC - Version
# MAGIC - Name (filename)
# MAGIC - Origin (mimetype, hash, filename)
# MAGIC - Furniture: hierarchical listing of headers / footers
# MAGIC - Body: hierarchical listing and ordering of body elements
# MAGIC - Groups: grouping of text (parent child)
# MAGIC
# MAGIC Parsed includes:
# MAGIC - Texts: Parsed text items
# MAGIC - Pictures: Parsed pictures with bounding boxes and images (if set)
# MAGIC - Tables: Parsed Tables with text and images (if set)
# MAGIC - Key_Value_Items: Lists with hierarchy
# MAGIC - Pages: Page images (if set)

# COMMAND ----------

result.document.dict()['pages'][1]

# COMMAND ----------

# MAGIC %md
# MAGIC We use docling to export all tables and images in a file based format to permit further downstream analysis, along with markdown files with linked and embedded images. Let's do this on one document and show the outputs in Unity.

# COMMAND ----------


