from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    IntegerType,
    BooleanType,
)

chunk_schema = StructType(
    [
        StructField("filename", StringType(), True),
        StructField("input_hash", StringType(), True),
        StructField("pages", ArrayType(IntegerType()), True),
        StructField("doc_refs", ArrayType(StringType()), True),
        StructField("has_table", BooleanType(), True),
        StructField("has_picture", BooleanType(), True),
        StructField("tables", ArrayType(StringType()), True),
        StructField("pictures", ArrayType(StringType()), True),
        StructField("headings", ArrayType(StringType()), True),
        StructField("captions", ArrayType(StringType()), True),
        StructField("chunk_type", StringType(), True),
        StructField("image_path", StringType(), True),
        StructField("text", StringType(), True),
        StructField("enriched_text", StringType(), True),
    ]
)

from docling_core.types.doc.labels import DocItemLabel
from docling.chunking import HybridChunker


def make_table_chunks(document, image_locations={}):
    input_hash = document.input_hash
    table_chunks = []
    for table in document.tables:
        pages = [x.page_no for x in table.prov]
        unique_pages = list(set(pages))
        unique_pages.sort()
        doc_refs = [table.self_ref] + [x.cref for x in table.children]
        headings = []
        captions = [table.caption_text(document)]
        tables = [table.self_ref.split("/")[-1]]
        pictures = []
        content_text = table.export_to_markdown()

        table_image_path = image_locations.get(table.self_ref.split("/")[-1], "")

        table_chunks.append(
            {
                "filename": document.origin.filename,
                "input_hash": input_hash,
                "pages": unique_pages,
                "doc_refs": doc_refs,
                "has_table": len(tables) > 0,
                "has_picture": len(pictures) > 0,
                "tables": tables,
                "pictures": pictures,
                "headings": headings,
                "captions": captions,
                "chunk_type": "table",
                "image_path": table_image_path,
                "text": content_text,
                "enriched_text": f"""
                Captions:{", ".join(captions) if captions else ""}\n
                Content:{content_text}
            """,
            }
        )

    return table_chunks


def make_picture_chunks(document, image_locations={}):
    input_hash = document.input_hash
    picture_chunks = []
    for pic in document.pictures:
        pages = [x.page_no for x in pic.prov]
        unique_pages = list(set(pages))
        unique_pages.sort()
        doc_refs = [pic.self_ref] + [x.cref for x in pic.children]
        headings = []
        captions = [pic.caption_text(document)]
        tables = []
        pictures = [pic.self_ref.split("/")[-1]]
        text = ", ".join([x.text for x in pic.annotations if x.kind == "description"])

        picture_image_path = image_locations.get(pic.self_ref.split("/")[-1], "")

        picture_chunks.append(
            {
                "filename": document.origin.filename,
                "input_hash": input_hash,
                "pages": unique_pages,
                "doc_refs": doc_refs,
                "has_table": len(tables) > 0,
                "has_picture": len(pictures) > 0,
                "tables": tables,
                "pictures": pictures,
                "headings": headings,
                "captions": captions,
                "chunk_type": "picture",
                "text": text,
                "image_path": picture_image_path,
                "enriched_text": f"""
                Captions:{", ".join(captions) if captions else ""}\n
                Text:{text}
            """,
            }
        )

    return picture_chunks


def make_page_chunks(document, image_locations={}):
    page_chunks = []

    for k, v in document.page_metadata.items():
        doc_refs = [
            item.self_ref
            for item, _level in document.iterate_items()
            if item.prov[0].page_no == k
        ]
        tables = [ref.split("/")[-1] for ref in doc_refs if ref.startswith("#/tables/")]
        pictures = [
            ref.split("/")[-1] for ref in doc_refs if ref.startswith("#/pictures/")
        ]
        headings = list(
            set(
                [
                    item.text
                    for item, _level in document.iterate_items()
                    if item.prov[0].page_no == k
                    and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
                ]
            )
        )
        captions = list(
            set(
                [
                    item.text
                    for item, _level in document.iterate_items()
                    if item.prov[0].page_no == k and item.label == DocItemLabel.CAPTION
                ]
            )
        )

        page_image_path = image_locations.get(k, "")

        page_chunk_dict = {
            "filename": document.origin.filename,
            "input_hash": document.input_hash,
            "pages": [k],
            "doc_refs": doc_refs,
            "has_table": len(tables) > 0,
            "has_picture": len(pictures) > 0,
            "tables": tables,
            "pictures": pictures,
            "headings": headings,
            "captions": captions,
            "chunk_type": "page",
            "text": v.text,
            "image_path": page_image_path,
            "enriched_text": f"""Headings:{", ".join(headings) if headings else ""}\nPage Description:{v.text}""",
        }
        page_chunks.append(page_chunk_dict)

    return page_chunks


def make_text_chunk(document, chunk):
    input_hash = document.input_hash
    pages = [[x.page_no for x in x.prov] for x in chunk.meta.doc_items]
    unique_pages = list(set([page for sublist in pages for page in sublist]))
    unique_pages.sort()
    doc_refs = [x.self_ref for x in chunk.meta.doc_items]
    headings = chunk.meta.headings
    captions = chunk.meta.captions
    tables = [
        x.self_ref.split("/")[-1]
        for x in chunk.meta.doc_items
        if x.label == DocItemLabel.TABLE
    ]
    pictures = [
        x.self_ref.split("/")[-1]
        for x in chunk.meta.doc_items
        if x.label == DocItemLabel.PICTURE
    ]

    return {
        "filename": document.origin.filename,
        "input_hash": input_hash,
        "pages": unique_pages,
        "doc_refs": doc_refs,
        "has_table": len(tables) > 0,
        "has_picture": len(pictures) > 0,
        "tables": tables,
        "pictures": pictures,
        "headings": headings,
        "captions": captions,
        "chunk_type": "text",
        "image_path": "",
        "text": chunk.text,
        "enriched_text": f"""
            Headings:{", ".join(headings) if headings else ""}\n
            Captions:{", ".join(captions) if captions else ""}\n
            Text:{chunk.text}
        """,
    }


def make_text_chunks(document, max_tokens=1000):
    chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
    chunks = list(chunker.chunk(document, max_tokens=max_tokens))
    chunk_dicts = [make_text_chunk(document, chunk) for chunk in chunks]
    return chunk_dicts


def chunk_maud_document(document, max_tokens=1000, image_locations={}):
    text_chunks = make_text_chunks(document, max_tokens=max_tokens)
    pic_chunks = make_picture_chunks(
        document, image_locations=image_locations.get("pictures", {})
    )
    table_chunks = make_table_chunks(
        document, image_locations=image_locations.get("tables", {})
    )
    page_chunks = make_page_chunks(
        document, image_locations=image_locations.get("pages", {})
    )
    return text_chunks + pic_chunks + table_chunks + page_chunks
