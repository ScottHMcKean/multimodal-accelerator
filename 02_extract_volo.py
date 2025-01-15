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
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.generate_table_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# COMMAND ----------

from pathlib import Path
test_path = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}/01-0571-0011_Rev06_07-07.pdf")
result = doc_converter.convert(test_path)

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
doc_name = test_path.stem

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

def capture_page_metadata(page, output_dir, doc_name, client):
  

# COMMAND ----------

# Extract page images and descriptions
from openai import OpenAI
# from dbmma.llm import get_open_ai_image_description

client = OpenAI(api_key = dbutils.secrets.get('shm','gpt4o'))

page_metadata = []
for _, page in result.document.pages.items():
    page_dir = output_dir / doc_name / 'pages'
    page_dir.mkdir(exist_ok=True, parents=True)

    page_image_path = page_dir / f"{page.page_no}.webp"
    
    description = get_open_ai_image_description(
      client, 
      page.image.pil_image, 
      image_type='page', 
      max_tokens=200
      )

    page_entry = {
      'output_dir': output_dir,
      'doc_name': doc_name,
      'page_no': page.page_no,
      'size': page.size,
      'img_path': str(page_image_path),
    }
    
    with page_image_path.open("wb") as fp:
        page.image.pil_image.save(fp, format="webp", quality=100)

    page_metadata.append(page_entry)

# COMMAND ----------

from pathlib import PosixPath
from pyspark.sql import Row

# Convert PosixPath to string
page_metadata = [
    Row(**{k: str(v) if isinstance(v, PosixPath) else v for k, v in row.items()})
    for row in page_metadata
]

# Create DataFrame
df = spark.createDataFrame(page_metadata)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC We use the rule of thumb that 1 tokens is ~ 0.75 words and prompt the model to limit the description to the same number of words as our max token limit for our chunks. This means that each table, page, and figure can be added as a single description for vector search.

# COMMAND ----------

from dbmma.llm import get_open_ai_image_description

get_open_ai_image_description(client, page.image.pil_image, image_type='page', max_tokens=200)

# COMMAND ----------

from dbmma.utils import encode_pil_image_to_base64
from openai import OpenAI
from PIL import Image



def get_open_ai_image_description(
    client: OpenAI, 
    image: Image.Image, 
    image_type:str='page', 
    max_tokens=200
    ):
    assert image_type in ['page', 'table', 'figure']
    
    img_bytes = encode_pil_image_to_base64(page.image.pil_image)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe the contents of this {image_type}. Keep the response within {max_tokens} words",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_bytes}"},
                    },
                ],
            }
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC We can also leverage language models to describe each page. To do this we use OpenAI GPT4o.

# COMMAND ----------

# Save page images
output_dir = Path(f'/Volumes/{CATALOG}/{SCHEMA}/{SILVER_PATH}')
doc_name = test_path.stem



table_dir = output_dir / doc_name / 'tables'
table_dir.mkdir(exist_ok=True, parents=True)

picture_dir = output_dir / doc_name / 'pictures'
picture_dir.mkdir(exist_ok=True, parents=True)



# Save images of figures and tables
table_counter = 0
picture_counter = 0
for element, _level in result.document.iterate_items():
    if isinstance(element, TableItem):
        table_counter += 1
        element_image_filename = table_dir / f"table-{table_counter}.png"
        with element_image_filename.open("wb") as fp:
            element.get_image(result.document).save(fp, "PNG")

    if isinstance(element, PictureItem):
        picture_counter += 1
        element_image_filename = picture_dir / f"picture-{picture_counter}.png"
        with element_image_filename.open("wb") as fp:
            element.get_image(result.document).save(fp, "PNG")

# Save markdown with embedded pictures (as binary)
md_filename = output_dir / doc_name / f"markdown-with-images.md"
result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

# Save markdown with externally referenced pictures
md_filename = output_dir / doc_name / f"markdown-with-image-refs.md"
result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

# COMMAND ----------


