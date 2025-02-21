class VLMEnrichmentPipelineOptions(PdfPipelineOptions):
    get_descriptions: bool = True
    classify_diagrams: bool = False


class VLMEnrichmentModel(BaseEnrichmentModel):
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            assert isinstance(element, PictureItem)

            # uncomment this to interactively visualize the image
            element.get_image(doc).show()

            element.annotations.append(
                PictureClassificationData(
                    provenance="example_classifier-0.0.1",
                    predicted_classes=[
                        PictureClassificationClass(class_name="dummy", confidence=0.42)
                    ],
                )
            )

            yield element


class ExamplePictureClassifierPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: ExamplePictureClassifierPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: ExamplePictureClassifierPipeline

        self.enrichment_pipe = [
            ExamplePictureClassifierEnrichmentModel(
                enabled=pipeline_options.do_picture_classifer
            )
        ]

    @classmethod
    def get_default_options(cls) -> ExamplePictureClassifierPipelineOptions:
        return ExamplePictureClassifierPipelineOptions()


# COMMAND ----------

filename = "92ab18cd-55d7-5c30-a347-2369dd751f74.pdf"
input_path = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}/{filename}")

# COMMAND ----------


doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=ExamplePictureClassifierPipeline,
            pipeline_options=pipeline_options,
        )
    }
)
result = doc_converter.convert(input_path)

for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        print(
            f"The model populated the `data` portion of picture {element.self_ref}:\n{element.annotations}"
        )

# COMMAND ----------

result

# COMMAND ----------

pdf_pipe_options

# COMMAND ----------

# MAGIC %md
# MAGIC This accelerator provides an abstraction for the converter module. Since converters are evolving so quickly and have important complexity, latency, cost, and throughput tradeoffs, no one converter is going to work for everyone. We therefor use a factory pattern
# MAGIC
# MAGIC This design provides several benefits:
# MAGIC - Unified Interface: The AbstractConverter class defines a common interface for all converters, with convert() and get_result() methods.
# MAGIC - Adapter Pattern: Each specific converter (Docling and Markitdown) is wrapped in an adapter class that implements the AbstractConverter interface.
# MAGIC - Encapsulation: The differences in the original converter interfaces and result formats are hidden behind the adapter classes.
# MAGIC - Factory Pattern: The ConverterFactory class provides a simple way to create the appropriate converter based on a type string.

# COMMAND ----------

from openai import OpenAI
from pathlib import Path
import time
from maud.meta_parsers import (
    save_page_metadata,
    save_table_metadata,
    save_picture_metadata,
)
from maud.converters import DoclingConverterAdapter

client = OpenAI(api_key=dbutils.secrets.get("shm", "gpt4o"))

# COMMAND ----------

# iterative on files from the bronze path
files = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}").glob("*")
types = [".xlsx", ".docx", ".pptx", ".pdf"]
filenames = [file.name for file in files if file.suffix in types]

iter_results = {}
for filename in filenames:
    start_time = time.time()
    print(filename)

input_path = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}/{filename}")
output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{SILVER_PATH}")
converter = DoclingConverterAdapter(
    input_path,
    output_dir,
    allowed_formats=docling_allowed_formats,
    format_options=docling_format_options,
)

# convert document
document = converter.convert()
converter.save_result()

# COMMAND ----------

doc_dict = document.model_dump()
doc_dict.keys()

# COMMAND ----------

document.model_extra()

# COMMAND ----------

# iterative on files from the bronze path
files = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}").glob("*")
types = [".xlsx", ".docx", ".pptx", ".pdf"]
filenames = [file.name for file in files if file.suffix in types]

iter_results = {}
for filename in filenames:
    start_time = time.time()
    print(filename)
    input_path = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{BRONZE_PATH}/{filename}")
    output_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{SILVER_PATH}")
    converter = DoclingConverterAdapter(
        input_path,
        output_dir,
        allowed_formats=docling_allowed_formats,
        format_options=docling_format_options,
    )

    # convert document
    document = converter.convert()

    converter.save_result()

    # get descriptions for pages, tables, and figures
    page_metadata = save_page_metadata(
        document, client, converter._output_path, converter._input_hash
    )
    table_metadata = save_table_metadata(
        document, client, converter._output_path, converter._input_hash
    )
    picture_metadata = save_picture_metadata(
        document, client, converter._output_path, converter._input_hash
    )

    result_summary = {
        "input_path": input_path,
        "input_hash": converter._input_hash,
        "output_dir": converter._output_path,
        "document": document,
        "page_meta": page_metadata,
        "table_meta": table_metadata,
        "picture_meta": picture_metadata,
    }
    iter_results[converter._input_hash] = result_summary
    print(f"time(s): {round(time.time() - start_time, 1)}")

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

page_meta_combined = []
for k, v in iter_results.items():
    page_meta_combined = page_meta_combined + v["page_meta"]

page_meta_df = pd.DataFrame(page_meta_combined)
page_meta_df

# COMMAND ----------

from maud.meta_parsers import page_meta_schema, table_meta_schema

page_meta = spark.createDataFrame(page_meta_df, page_meta_schema)
(
    page_meta.write.mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable("shm.multimodal.page_metadata")
)
display(page_meta)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables
# MAGIC We now move on to tables - trying to ensure that we match the order of the tables within the document so we can trace them back properly.

# COMMAND ----------

table_meta_combined = []
for k, v in iter_results.items():
    table_meta_combined = table_meta_combined + v["table_meta"]

table_meta_df = pd.DataFrame(table_metadata)
table_meta_df

# COMMAND ----------

table_meta = spark.createDataFrame(table_meta_df, table_meta_schema)
(
    table_meta.write.mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable("shm.multimodal.table_metadata")
)
display(table_meta)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pictures

# COMMAND ----------

pic_meta_combined = []
for k, v in iter_results.items():
    pic_meta_combined = pic_meta_combined + v.get("picture_meta", [])

pic_meta_df = pd.DataFrame(pic_meta_combined)
pic_meta_df

# COMMAND ----------

if pic_meta_combined is not None:
    pic_meta = spark.createDataFrame(pic_meta_df, table_meta_schema)
    (
        pic_meta.write.mode("overwrite")
        .option("mergeSchema", "true")
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

from docling.chunking import HybridChunker, HierarchicalChunker
from maud.chunkers import process_chunk

chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")

combined_processed_chunks = []
for key, result in iter_results.items():
    chunk_iter = chunker.chunk(dl_doc=result["document"], max_tokens=200)
    processed_chunks = []
    for chunk in chunk_iter:
        try:
            processed_chunks.append(process_chunk(chunk.model_dump()))
        except:
            continue
    combined_processed_chunks = combined_processed_chunks + processed_chunks

# COMMAND ----------

# MAGIC %md
# MAGIC The hybrid chunker seems to do a good job of keep heading and page based separations. Pages with limited or no text still maintain seperate chunks. Also, it keeps metadata about headings, captions, etc. (see chunk 57), as well as image and table content (see chunk 24). This makes the resultant outputs after parsing huge, but they are all there.

# COMMAND ----------

# Define the schema
from maud.document.chunkers import chunk_schema

chunk_df = spark.createDataFrame(combined_processed_chunks, schema=chunk_schema)
(chunk_df.write.mode("overwrite").saveAsTable("shm.multimodal.processed_chunks"))
display(chunk_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We have context aware chunks with metadata, a full markdown representation, and the ability to store all of those in volumes or a delta table. We now move on to featurization to build a better representation of the document using the ordering established by docling, the text chunks, and the image, table, and page metadata.
