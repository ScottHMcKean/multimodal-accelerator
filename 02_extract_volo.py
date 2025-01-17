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

import pandas as pd
from pathlib import Path
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, MapType, FloatType, DoubleType
from importlib import reload
import maud
reload(maud)

# COMMAND ----------

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import DoclingDocument

# setup the conversion pipeline to extract images and tables automatically
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.generate_table_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC Here we convert a single document using Docling. We are working on scaling and benchmarking this against other tools.

# COMMAND ----------

doc_path = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}/01-0571-0011_Rev06_07-07.pdf")
result = doc_converter.convert(doc_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volo Tests on Speed Up:
# MAGIC 1. Image_Resolution_Scale: Higher number - higher resolution. Need to test whether we need higher resolution for images when using MLLM.
# MAGIC Testing done on the 01-0571-0011_Rev06_07-07.pdf
# MAGIC - When set at 2: 59s
# MAGIC - When set at 1: 56s
# MAGIC - When set at 0.5: 54s
# MAGIC 2. Parallelize over Multiple Documents

# COMMAND ----------

import time
# Step 1
IMAGE_RESOLUTION_SCALE = 2

# Step 1
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.generate_table_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

pdf_folder = f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}"
all_pdfs = [str(p) for p in Path(pdf_folder).glob("*.pdf")]
print("PDFs found:", all_pdfs)

results = {}
global_start_time = time.time()  # Track total processing time

for pdf in all_pdfs:
    start_time = time.time()
    result = doc_converter.convert(pdf)
    end_time = time.time()
    
    # Measure how long this PDF took
    duration = end_time - start_time
    
    # Attempt to fetch the number of pages from the result
    # Assuming docling returns a DoclingDocument or similar with a `.pages` attribute
    # Adjust as needed if the pages are stored differently in your result object.
    try:
        num_pages = len(result.pages)
    except AttributeError:
        # Fallback if there's no .pages attribute
        num_pages = 0

    results[pdf] = result
    
    print(f"Processed {pdf}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Number of pages: {num_pages}")

global_end_time = time.time()
total_processing_time = global_end_time - global_start_time
print(f"Total processing time: {total_processing_time:.2f} seconds")

# COMMAND ----------

#Step 2:
import time
from pathlib import Path
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

pdf_folder = f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}"
all_pdfs = [str(p) for p in Path(pdf_folder).glob("*.pdf")]
print("Found PDFs:", all_pdfs)

rdd = spark.sparkContext.parallelize(all_pdfs, numSlices=8)

def convert_with_docling(pdf_path_str):
    import time
    from pathlib import Path
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    
    # Record start time inside the worker
    start_time = time.time()
    
    # Configure Docling pipeline inside each worker
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    path_obj = Path(pdf_path_str)
    try:
        # Perform conversion
        docling_result = doc_converter.convert(path_obj)
        
        # End time + duration
        end_time = time.time()
        duration = end_time - start_time

        # Attempt to get page count (adjust if docling_result exposes pages differently)
        if hasattr(docling_result, 'pages'):
            num_pages = len(docling_result.pages)
        else:
            num_pages = 0

        # Return all info back to the driver
        return (pdf_path_str, "success", duration, num_pages)

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        return (pdf_path_str, f"error: {str(e)}", duration, 0)


# Measure the total time on the driver
global_start = time.time()
results = rdd.map(convert_with_docling).collect()
global_end = time.time()
total_time = global_end - global_start

# Print info for each PDF
for pdf_path_str, status, duration, num_pages in results:
    if status == "success":
        print(f"Processed {pdf_path_str}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Number of pages: {num_pages}")
    else:
        print(f"Failed to convert {pdf_path_str}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Error: {status}")

# Finally, print total processing time
print(f"Total processing time: {total_time:.2f} seconds")

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

# MAGIC %md
# MAGIC We want to cache the results along with the location and save it to our parse table. We can also scale this into tables to help with scaling. We save the document as markdown because it is extensible, readable by multiple formats, and seems to be the way the industry is moving.

# COMMAND ----------

# Save document as markdown
output_dir = Path(f'/Volumes/{CATALOG}/{SCHEMA}/{SILVER_PATH}')
doc_name = doc_path.stem

result.document.save_as_markdown(
  output_dir / doc_name / f"markdown-with-image-refs.md", 
  image_mode=ImageRefMode.REFERENCED
  )
  
result.document.save_as_markdown(
  output_dir / doc_name / f"markdown-with-images.md", 
  image_mode=ImageRefMode.EMBEDDED
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are going to extract labelled pages, images, and tables for further metadata capture. We want to be systematic here so we can reference when we move to chunking and retrieval, which is why we want to use a framework like Docling. We are going to save each item as a file, but capture the location and metadata in a table.
# MAGIC
# MAGIC We use the webp format because it is smaller and less lossy compared to png and jpeg, which should improve our app performance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pages
# MAGIC
# MAGIC We can also leverage language models to describe each page. To do this we use OpenAI GPT4o.
# MAGIC We use the rule of thumb that 1 tokens is ~ 0.75 words and prompt the model to limit the description to the same number of words as our max token limit for our chunks. This means that each table, page, and figure can be added as a single description for vector search.

# COMMAND ----------

# Extract page images and descriptions
from openai import OpenAI
from maud.metadata import capture_page_metadata, save_page_image

client = OpenAI(api_key = dbutils.secrets.get('shm','gpt4o'))

page_metadata = []
page_dir = output_dir / doc_name / 'pages'
page_dir.mkdir(exist_ok=True, parents=True)

for _, page in list(result.document.pages.items())[:2]:
    page_entry = capture_page_metadata(page, page_dir, doc_name, client)
    page_metadata.append(page_entry)
    save_page_image(page, page_dir)

# COMMAND ----------

page_meta_df = pd.DataFrame(page_metadata)
page_meta_df

# COMMAND ----------

page_meta_schema = StructType([
    StructField("doc_name", StringType(), False),
    StructField("ref", StringType(), False),
    StructField("type", StringType(), True),
    StructField("page_no", IntegerType(), True),
    StructField("size", StructType([
        StructField("width", DoubleType(), True),
        StructField("height", DoubleType(), True)
    ]), True),
    StructField("img_path", StringType(), True),
    StructField("description", StringType(), True)
])

page_meta = spark.createDataFrame(page_meta_df, page_meta_schema)
(
  page_meta.write
  .mode('overwrite')
  .option("mergeSchema", "true")
  .saveAsTable("shm.multimodal.page_metadata")
)
display(page_meta)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables
# MAGIC We now move on to tables - trying to ensure that we match the order of the tables within the document so we can trace them back properly. 

# COMMAND ----------

from maud.metadata import capture_table_metadata, save_table_image

table_metadata = []
table_dir = output_dir / doc_name / 'tables'
table_dir.mkdir(exist_ok=True, parents=True)

for table in result.document.tables[0:3]:
    table_entry = capture_table_metadata(table, output_dir, doc_name, client)
    table_metadata.append(table_entry)
    save_table_image(table, output_dir)

# COMMAND ----------

table_meta_df = pd.DataFrame(table_metadata)
table_meta_df 

# COMMAND ----------

table_meta_schema = StructType([
    StructField("doc_name", StringType(), False),
    StructField("ref", StringType(), False),
    StructField("type", StringType(), True),
    StructField("parent", StringType(), True),
    StructField("page_no", IntegerType(), True),
    StructField("bbox", StructType([
        StructField("l", DoubleType(), True),
        StructField("t", DoubleType(), True),
        StructField("r", DoubleType(), True),
        StructField("b", DoubleType(), True)
    ]), True),
    StructField("caption_ref", StringType(), True),
    StructField("caption_index", IntegerType(), True),
    StructField("caption_text", StringType(), True),
    StructField("img_path", StringType(), True),
    StructField("description", StringType(), True)
])

table_meta = spark.createDataFrame(table_meta_df, table_meta_schema)
(
  table_meta.write
  .mode('overwrite')
  .saveAsTable("shm.multimodal.table_metadata")
)
display(table_meta)

# COMMAND ----------

from maud.metadata import capture_picture_metadata, save_picture_image

pic_metadata = []
pic_dir = output_dir / doc_name / 'pictures'
pic_dir.mkdir(exist_ok=True, parents=True)

for picture in result.document.pictures[0:3]:
    pic_entry = capture_picture_metadata(picture, pic_dir, doc_name, client)
    pic_metadata.append(pic_entry)
    save_picture_image(picture, pic_dir)

# COMMAND ----------

pic_meta_df = pd.DataFrame(pic_metadata)
pic_meta_df

# COMMAND ----------

pic_meta_schema = StructType([
    StructField("doc_name", StringType(), False),
    StructField("ref", StringType(), False),
    StructField("type", StringType(), True),
    StructField("parent", StringType(), True),
    StructField("page_no", IntegerType(), True),
    StructField("bbox", StructType([
        StructField("l", DoubleType(), True),
        StructField("t", DoubleType(), True),
        StructField("r", DoubleType(), True),
        StructField("b", DoubleType(), True)
    ]), True),
    StructField("caption_ref", StringType(), True),
    StructField("caption_index", IntegerType(), True),
    StructField("caption_text", StringType(), True),
    StructField("img_path", StringType(), True),
    StructField("description", StringType(), True)
])

pic_meta = spark.createDataFrame(pic_meta_df, pic_meta_schema)
(
  pic_meta.write
  .mode('overwrite')
  .saveAsTable("shm.multimodal.picture_metadata")
)
display(pic_meta)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Chunking
# MAGIC Docling does an excellent job of organizing text. It also provides a hierarchical chunker we can use to prepare the bulk of our document for vector search. We will then take the page, table, and picture metadata with descriptions and augment the standard text information for vector search.
# MAGIC
# MAGIC We now have a fully extracted DoclingDocument that can be used for downstream analysis. Let's use the hierarchical parser to chunk one of the documents in the table and compare against the markdown format. We will load a MiniLM tokenizer to control the max tokens.

# COMMAND ----------

from docling.chunking import HybridChunker

chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
chunk_iter = chunker.chunk(
  dl_doc=result.document,
  max_tokens=200)
chunks = list(chunk_iter)

# COMMAND ----------

# MAGIC %md
# MAGIC The hybrid chunker seems to do a good job of keep heading and page based separations. Pages with limited or no text still maintain seperate chunks. Also, it keeps metadata about headings, captions, etc. (see chunk 57), as well as image and table content (see chunk 24). This makes the resultant outputs after parsing huge, but they are all there.

# COMMAND ----------

from maud.chunk import process_chunk
processed_chunks = [process_chunk(chunk.model_dump()) for chunk in chunks]
processed_chunks[0]

# COMMAND ----------

# Define the schema
chunk_schema = StructType([
    StructField('text', StringType(), True),
    StructField('filename', StringType(), True),
    StructField('heading', StringType(), True),
    StructField('caption', StringType(), True),
    StructField('items', ArrayType(MapType(StringType(), StringType())), True),
    StructField('parents', ArrayType(StringType()), True),
    StructField('pages', ArrayType(IntegerType()), True)
])

chunk_df = spark.createDataFrame(processed_chunks, schema=chunk_schema)
(
  chunk_df.write
  .mode('overwrite')
  .saveAsTable("shm.multimodal.processed_chunks")
)
display(chunk_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC We have context aware chunks with metadata, a full markdown representation, and the ability to store all of those in volumes or a delta table. We now move on to featurization to build a better representation of the document using the ordering established by docling, the text chunks, and the image, table, and page metadata. 

# COMMAND ----------


