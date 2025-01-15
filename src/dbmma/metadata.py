from .llm import get_open_ai_image_description

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
    
    description = get_open_ai_image_description(
      client, 
      page.image.pil_image, 
      image_type='page', 
      max_tokens=200
      )

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
    
    description = get_open_ai_image_description(
      client, 
      table.image.pil_image, 
      image_type='table', 
      max_tokens=200
      )

    try:
        cap_ref = table.captions[0].cref
    except:
        cap_ref = ""

    try:
        cap_index = int(table.captions[0].cref.split("/")[-1])
    except:
        cap_index = -1

    try:
        cap_text = result.document.texts[int(table.captions[0].cref.split("/")[-1])].text
    except:
        cap_text = ""

    table_entry = {
        "doc_name": doc_name,
        "ref": table.self_ref,
        "type": "table",
        "parent": str(table.parent),
        "page_no": table.prov[0].page_no,
        "bbox": clean_bbox(dict(table.prov[0].bbox)),
        "caption_ref": cap_ref,
        "caption_index": cap_index,
        "caption_text": cap_text,
        'img_path': str(table_image_path),
        "description": description
    }
        
    return table_entry
  

def capture_picture_metadata(picture, pic_dir, doc_name, client):
    """
    Parse docling picture metadata for vector search
    """

    pic_image_path = pic_dir / f"{picture.prov[0].page_no}.webp"
    
    description = get_open_ai_image_description(
      client, 
      picture.image.pil_image, 
      image_type='picture', 
      max_tokens=200
      )

    try:
        cap_ref = picture.captions[0].cref
    except:
        cap_ref = ""

    try:
        cap_index = int(picture.captions[0].cref.split("/")[-1])
    except:
        cap_index = -1

    try:
        cap_text = result.document.texts[int(picture.captions[0].cref.split("/")[-1])].text
    except:
        cap_text = ""

    table_entry = {
        "doc_name": doc_name,
        "ref": picture.self_ref,
        "type": "picture",
        "parent": str(picture.parent),
        "page_no": picture.prov[0].page_no,
        "bbox": clean_bbox(dict(picture.prov[0].bbox)),
        "caption_ref": cap_ref,
        "caption_index": cap_index,
        "caption_text": cap_text,
        "img_path": str(pic_image_path),
        "description": description
    }
        
    return table_entry


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
