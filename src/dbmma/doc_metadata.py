from .llm import get_open_ai_image_description

def capture_page_metadata(page, output_dir, doc_name, client):
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
      'doc_name': doc_name,
      'page_no': page.page_no,
      'size': page.size,
      'img_path': str(page_image_path),
    }
    
    return page_entry

def capture_table_metadata(table, output_dir, doc_name, client):
    table_dir = output_dir / doc_name / 'tables'
    table_dir.mkdir(exist_ok=True, parents=True)

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
        cap_index = ""

    try:
        cap_text = result.document.texts[int(table.captions[0].cref.split("/")[-1])].text
    except:
        cap_text = ""

    table_entry = {
        "doc_name": doc_name,
        "table_ref": table.self_ref,
        "parent": table.parent,
        "page_no": table.prov[0].page_no,
        "bbox": dict(table.prov[0].bbox),
        "caption_ref": cap_ref,
        "caption_index": cap_index,
        "caption_text": cap_text,
        "description": description
    }
        
    return table_entry
  

def save_page_image(page, output_dir):
    page_image_path = output_dir / f"{page.page_no}.webp"
    with page_image_path.open("wb") as fp:
        page.image.pil_image.save(fp, format="webp", quality=100)

def save_table_image(table, output_dir):
    table_image_path = output_dir / f"{table.self_ref.split('/')[-1]}.webp"
    with table_image_path.open("wb") as fp:
        table.image.pil_image.save(fp, format="webp", quality=100)

def save_picture_image(picture, output_dir):
    picture_image_path = output_dir / f"{picture.self_ref.split('/')[-1]}.webp"
    with table_image_path.open("wb") as fp:
        table.image.pil_image.save(fp, format="webp", quality=100)
