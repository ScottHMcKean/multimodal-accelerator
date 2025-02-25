# MAUD LangGraph Documentation

MAUD uses the LangGraph library to build and deploy agent workflows. This documentation is a summary of the LangGraph documentation and the code in the repository.

The [graph definitions](https://langchain-ai.github.io/langgraph/reference/graphs/#graph-definitions) in the LangGraph documentation are helpful for understanding the options for constructing workflows.

LangGraph offers multiple streaming modes, but this repo uses the basic [stream](https://langchain-ai.github.io/langgraph/concepts/streaming/#streaming) mode since MLflow doesn't support async. This allows returning node outputs to users during execution, enabling feedback during intermediate steps.

## Deployment Models

We use MLFLow's [models from code](https://mlflow.org/docs/latest/model/models-from-code.html) approach to logging. There are some [limitations to be aware of](https://mlflow.org/docs/latest/model/dependencies.html#caveats-of-code-paths-option) that matter for how the project is structure.

LangGraph models can be deployed using MLFlow's [langchain flavor](https://mlflow.org/docs/latest/python_api/mlflow.langchain.html#mlflow.langchain.log_model) or using pyfunc and the [ChatModel](https://mlflow.org/docs/latest/llms/chat-model-intro/index.html) interface.

## Signatures

See Databricks [documentation](https://docs.databricks.com/en/generative-ai/agent-framework/agent-schema.html#define-an-agents-input-and-output-schema) on agent input and output schema.

LangGraph applications require a post processing step to format the graph's state (a dictionary) into an MLflow chat model response type. For batch inference, the response type is [ChatCompletionResponse](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatCompletionResponse); for streaming, yielding instances of [ChatCompletionChunk](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.llm.ChatCompletionChunk) is required.

## Limitations

- The predict_stream() method works in the Notebook for streaming graph events, it was not supported by serverless inference at the time this nobtebook was developed.
- Directory structure matters, so be cautious in changing it too much for both asset bundles and model logging. 