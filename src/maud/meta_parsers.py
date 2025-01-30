from .foundation_models import get_open_ai_image_description
from pathlib import Path
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

def clean_bbox(bbox_entry):
    """
    Clean the types out of bbox and make it a simple dict
    """
    return {k: float(v) for k, v in bbox_entry.items() if k != 'coord_origin'}

def capture_page_metadata(page, page_dir, doc_name, client):
    """
    Parse docling page metadata for vector search
    """

    page_image_path = page_dir / f"{page.page_no}.webp"
    
    try:
        description = get_open_ai_image_description(
        client, 
        page.image.pil_image, 
        image_type='page', 
        max_tokens=200
        )
    except:
        description = ""

    page_entry = {
      'doc_name': doc_name,
      'ref': f"#/page/{page.page_no}",
      'type': 'page',
      'page_no': page.page_no,
      'size': dict(page.size),
      'img_path': str(page_image_path),
      'description': description
    }
    
    return page_entry

def capture_table_metadata(table, table_dir, doc_name, client):
    """
    Parse docling table metadata for vector search
    """

    table_image_path = table_dir / f"{table.prov[0].page_no}.webp"

    try:    
        description = get_open_ai_image_description(
        client, 
        table.image.pil_image, 
        image_type='table', 
        max_tokens=200
        )
    except:
        description = ""

    try:
        cap_ref = table.captions[0].cref
    except:
        cap_ref = ""

    try:
        cap_index = int(table.captions[0].cref.split("/")[-1])
    except:
        cap_index = -1

    table_entry = {
        "doc_name": doc_name,
        "ref": table.self_ref,
        "type": "table",
        "parent": str(table.parent),
        "page_no": table.prov[0].page_no,
        "bbox": clean_bbox(dict(table.prov[0].bbox)),
        "caption_ref": cap_ref,
        "caption_index": cap_index,
        'img_path': str(table_image_path),
        "description": description
    }
        
    return table_entry
  

def capture_picture_metadata(picture, pic_dir, doc_name, client):
    """
    Parse docling picture metadata for vector search
    """

    pic_image_path = pic_dir / f"{picture.prov[0].page_no}.webp"
    
    try:
        description = get_open_ai_image_description(
            client, 
            picture.image.pil_image, 
            image_type='picture', 
            max_tokens=200
            )
    except:
        description = ""

    try:
        cap_ref = picture.captions[0].cref
    except:
        cap_ref = ""

    try:
        cap_index = int(picture.captions[0].cref.split("/")[-1])
    except:
        cap_index = -1

    picture_entry = {
        "doc_name": doc_name,
        "ref": picture.self_ref,
        "type": "picture",
        "parent": str(picture.parent),
        "page_no": picture.prov[0].page_no,
        "bbox": clean_bbox(dict(picture.prov[0].bbox)),
        "caption_ref": cap_ref,
        "caption_index": cap_index,
        "img_path": str(pic_image_path),
        "description": description
    }
        
    return picture_entry


def save_page_image(page, page_dir):
    page_image_path = page_dir / f"{page.page_no}.webp"
    with page_image_path.open("wb") as fp:
        page.image.pil_image.save(fp, format="webp", quality=100)

def save_table_image(table, table_dir):
    table_image_path = table_dir / f"{table.self_ref.split('/')[-1]}.webp"
    with table_image_path.open("wb") as fp:
        table.image.pil_image.save(fp, format="webp", quality=100)

def save_picture_image(picture, pic_dir):
    picture_image_path = pic_dir / f"{picture.self_ref.split('/')[-1]}.webp"
    with picture_image_path.open("wb") as fp:
        picture.image.pil_image.save(fp, format="webp", quality=100)

def save_page_metadata(document, client, output_path: Path, file_hash_key, limit=5):
    page_metadata = []
    page_dir = output_path / 'pages'
    page_dir.mkdir(exist_ok=True, parents=True)

    loaded_pages = 0
    for _, page in list(document.pages.items()):
        page_entry = capture_page_metadata(page, page_dir, file_hash_key, client)
        page_metadata.append(page_entry)
        if page.image is not None:
            save_page_image(page, page_dir)
        loaded_pages += 1
        
        if loaded_pages > limit:
            break

    return page_metadata

def save_table_metadata(document, client, output_path: Path, file_hash_key, limit=5):
    table_metadata = []
    table_dir = output_path / 'tables'
    table_dir.mkdir(exist_ok=True, parents=True)

    loaded_tables = 0
    for table in document.tables:
        table_entry = capture_table_metadata(table, table_dir, file_hash_key, client)
        table_metadata.append(table_entry)
        if table.image is not None:
            save_table_image(table, table_dir)
        loaded_tables += 1
        
        if loaded_tables > limit:
            break

    return table_metadata

def save_picture_metadata(document, client, output_path: Path, file_hash_key, limit=5):
    pic_metadata = []
    pic_dir = output_path / 'pictures'
    pic_dir.mkdir(exist_ok=True, parents=True)

    loaded_pictures = 0
    for picture in document.pictures:
        pic_entry = capture_picture_metadata(picture, pic_dir, file_hash_key, client)
        pic_metadata.append(pic_entry)
        if picture.image is not None:
            save_picture_image(picture, pic_dir)
        loaded_pictures += 1
        
        if loaded_pictures > limit:
            break

    return pic_metadata

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
    StructField("img_path", StringType(), True),
    StructField("description", StringType(), True)
])

