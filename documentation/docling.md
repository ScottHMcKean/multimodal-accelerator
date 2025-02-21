# MAUD Docling Documentation
MAUD extends the base Docling library to do additional metadata enrichment. There are two main processes in a docling converter pipeline - the build and the enrichment. We extend both in order to classify and describe pages and elements (figures and tables) within a document. The overally goal here is to enable production implementation of hierarchy and knowledge graph approaches to retrieval.

## Build

The build process does the conversion using the OCR engine.

## Enrichment

The enrichment process loops through elements in the document (after the build process) and adds additional metadata. Internally, docling has an enrichment pipeline build for pictures, but MAUD extends this to tables and pages. This results in three distinct pipelines - page, table, and picture descriptions. 

### Pages
We had to extend the document class to include a `page_metadata` object. Once the document build has completed we add the page metadata using an LLM description. The intent is to extend this to entity recognition in the future to build up knowledge graphs. We use the PictureData class for pages, since it is built into the library. 

### Pictures
We use the annotations field in the PictureItem class that leverages the PictureDescriptionData and PictureClassificationData classes. The output of the custom annotations looks like this:

```
'annotations': [
    {
        'kind': 'classification',
         'provenance': 'example_classifier-0.0.1',
         'predicted_classes': [{'class_name': 'dummy', 'confidence': 0.42}]
    },
    {
        'kind': 'description',
        'text': 'This is a dummy description',
        'provenance': 'a llm model'
    }
]
```

### Tables
We use the captions field in the TableItem class to store descriptions. First we create a text reference item, then reference that via the caption list which is expecting a reference item and not plain text.


## Limitations

- Docling currently does not support .ppt, .doc, and .xls files. We can hack a solution for this, but decided not to support these formats for now.