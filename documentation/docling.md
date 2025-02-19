# MAUD Docling Documentation

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