import os
import tempfile
from pathlib import Path
from mlflow.pyfunc import PythonModel
from typing import List
import mlflow

from databricks.sdk import WorkspaceClient

class DoclingModel(PythonModel):
    def __init__(self):
        # fix for opencv bug
        import os

        os.environ.pop("OPENSSL_FORCE_FIPS_MODE", None)

        self.client = None
        self.converter = None
        self.output_volume_root = "/Volumes/shm/multimodal/exports"

    def initialize_agent(self):
        # service principal authorization
        self.w = WorkspaceClient(
            host=os.environ["DATABRICKS_HOST"],
            client_id=os.environ["DATABRICKS_CLIENT_ID"],
            client_secret=os.environ["DATABRICKS_CLIENT_SECRET"],
        )

    def load_context(self, context):
        print("loading_context")
        # load this with context since it triggers library installs
        from docling.document_converter import DocumentConverter
        self.converter = DocumentConverter()

    def predict(self, model_input: List[str], params=None) -> List[str]:
        if self.converter is None:
            print("Load context to enable converter")
            return model_input

        print("initialize agent")
        self.initialize_agent()

        output_paths = []

        for input_path in model_input:
            file_stem = Path(input_path).stem

            # Download the file from the volume to a local temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                self.w.files.download_to(input_path, tmpfile.name)
                local_input_path = tmpfile.name

            result = self.converter.convert(local_input_path)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = f"{tmpdir}/{file_stem}.md"
                markdown_content = result.document.export_to_markdown(temp_path)

                # Write the markdown content to the file
                with open(temp_path, "w") as f:
                    f.write(markdown_content)

                # Upload to Volumes using Databricks SDK
                volume_path = f"{self.output_volume_root}/{file_stem}.md"

                with open(temp_path, "r") as f:
                    file_data = f.read()

                self.w.dbutils.fs.put(
                    file=volume_path,
                    contents=file_data,
                    overwrite=True,
                )

            output_paths.append(volume_path)

        return output_paths

model = DoclingModel()
mlflow.models.set_model(model)