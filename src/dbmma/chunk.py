from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, MapType, FloatType, DoubleType

def get_unique_entries(input_list):
    return list(set(input_list))
  
def process_chunk_headings(chunk):
    headings = chunk['meta'].get('headings', [''])
    if headings is None:
        return ""
    
    return (" ").join(headings)  

def process_chunk_captions(chunk):
    captions = chunk['meta'].get('captions', [''])
    if captions is None:
        return ""
    
    return (" ").join(captions)  

def process_chunk_parents(chunk):
    return list(set([x['parent']['cref'] for x in chunk['meta']['doc_items']]))

def process_chunk_pages(chunk):
    return list(set([x['prov'][0]['page_no'] for x in chunk['meta']['doc_items']]))

def process_chunk(chunk):
    return {
        'text': chunk['text'],
        'filename': chunk['meta']['origin']['filename'],
        'heading': process_chunk_headings(chunk),
        'caption': process_chunk_captions(chunk),
        'ref': [x['self_ref'] for x in chunk['meta']['doc_items']],
        'items': [{'ref':x['self_ref'],'label':str(x['label'])} for x in chunk['meta']['doc_items']],
        'parents': process_chunk_parents(chunk),
        'pages': process_chunk_pages(chunk)
    }

chunk_schema = StructType([
    StructField('text', StringType(), True),
    StructField('filename', StringType(), True),
    StructField('heading', StringType(), True),
    StructField('caption', StringType(), True),
    StructField('items', ArrayType(MapType(StringType(), StringType())), True),
    StructField('parents', ArrayType(StringType()), True),
    StructField('pages', ArrayType(IntegerType()), True)
])