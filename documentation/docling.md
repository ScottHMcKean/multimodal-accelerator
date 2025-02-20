# MAUD Docling Documentation
MAUD extends the base Docling library to do additional metadata enrichment. There are two main processes in a docling converter pipeline - the build and the enrichment. We extend both in order to classify and describe pages and elements (figures and tables) within a document. The overally goal here is to enable production implementation of hierarchy and knowledge graph approaches to retrieval.

## Build

The build process does the conversion using the OCR engine.

## Enrichment

The enrichment process loops through elements in the document (after conversion) and adds additional metadata. The best built out enrichment pipeline right now is for figures (or Pictures in Docling terms).

## Custom Annotations

We use the annotations field in the PictureItem and TableItem classes to store custom annotations. For figures, we use the PictureDescriptionData and PictureClassificationData classes. For tables, we use the TableDescriptionData and TableClassificationData classes.

This permits the incorporation of two distinct custom pipelines for tables, figures, and pages: a classification pipeline and a description pipeline.

The output of the model looks a bit like this:

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

## Limitations

- Docling currently does not support .ppt, .doc, and .xls files. We can hack a solution for this, but decided not to support these formats for now.