from databricks.sdk import WorkspaceClient
from openai import OpenAI
import openai
from typing import List, Optional
from mlflow.entities import Document
import mlflow
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams
from dataclasses import asdict
import json
import backoff
import mlflow
import os
from maud.agent.config import parse_config, MaudConfig


class FunctionCallingAgent(mlflow.pyfunc.ChatModel):
    """
    Retriever Calling Agent
    """

    def __init__(self, config: MaudConfig = None):
        """
        Initialize the OpenAI SDK client connected to Model Serving.
        Load the Agent's configuration from MLflow Model Config.
        """
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            self.mlflow_config = mlflow.models.ModelConfig(
                development_config=config_path
            )
            self.maud_config = parse_config(self.mlflow_config)
        else:
            self.maud_config = config

        self.workspace_client = WorkspaceClient()
        self.llm_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )

        # Vector Search
        mlflow.models.set_retriever_schema(
            name=self.maud_config.retriever.index_name,
            primary_key=self.maud_config.retriever.mapping.primary_key,
            text_column=self.maud_config.retriever.mapping.chunk_text,
            doc_uri=self.maud_config.retriever.mapping.document_uri,
        )

        # OpenAI-formatted function for the retriever tool
        self.retriever_tool_spec = [
            {
                "type": "function",
                "function": {
                    "name": self.maud_config.retriever.tool_name,
                    "description": self.maud_config.retriever.tool_description,
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "additionalProperties": False,
                        "properties": {
                            "query": {
                                "description": "query to look up in retriever",
                                "type": "string",
                            }
                        },
                    },
                },
            }
        ]

        # Identify the function used as the retriever tool
        self.tool_functions = {self.maud_config.retriever.tool_name: self.retrieve_docs}

    @mlflow.trace(name="rag_agent", span_type="AGENT")
    def predict(
        self,
        context=None,
        messages: List[ChatMessage] = None,
        params: Optional[ChatParams] = None,
    ) -> ChatCompletionResponse:
        """
        Primary function that takes a user's request and generates a response.
        """
        if messages is None:
            raise ValueError("predict(...) called without `messages` parameter.")

        # Convert all input messages to dict from ChatMessage
        messages = convert_chat_messages_to_dict(messages)

        # Add system prompt
        request = {
            "messages": [
                {"role": "system", "content": self.maud_config.agent.system_prompt},
                *messages,
            ],
            "tool_choice": "required",
        }

        # Ask the LLM to call tools & generate the response
        output = self.recursively_call_and_run_tools(**request)

        # Convert response to ChatCompletionResponse dataclass
        return ChatCompletionResponse.from_dict(output)

    @mlflow.trace(span_type="RETRIEVER", name="vector_search_retriever")
    def retrieve_docs(self, query: str) -> List[dict]:
        """
        Performs vector search to retrieve relevant chunks.

        Args:
            query: Search query.
            filters: Optional filters to apply to the search. Should follow the LLM-generated filter pattern of a list of field/filter pairs that will be converted to Databricks Vector Search filter format.

        Returns:
            List of retrieved Documents.
        """
        traced_search = mlflow.trace(
            self.workspace_client.vector_search_indexes.query_index,
            name="_workspace_client.vector_search_indexes.query_index",
            span_type="FUNCTION",
        )

        results = traced_search(
            index_name=self.maud_config.retriever.index_name,
            query_text=query,
            columns=self.maud_config.retriever.mapping.all_columns,
            **self.maud_config.retriever.parameters,
        )

        # We turn the config into a dict and pass it here

        return self.convert_vector_search_to_documents(
            results.as_dict(), self.maud_config.retriever.score_threshold
        )

    @mlflow.trace(span_type="PARSER")
    def convert_vector_search_to_documents(
        self, vs_results, vector_search_threshold
    ) -> List[dict]:

        column_names = []
        for column in vs_results["manifest"]["columns"]:
            column_names.append(column)

        docs = []
        if vs_results["result"]["row_count"] > 0:
            for item in vs_results["result"]["data_array"]:
                metadata = {}
                score = item[-1]
                if score >= vector_search_threshold:
                    metadata["similarity_score"] = score
                    for i, field in enumerate(item[0:-1]):
                        metadata[column_names[i]["name"]] = field
                    # put contents of the chunk into page_content
                    text_col_name = self.maud_config.retriever.mapping.chunk_text
                    page_content = metadata[text_col_name]
                    del metadata[text_col_name]

                    # put the primary key into id
                    id_col_name = self.maud_config.retriever.mapping.primary_key
                    id = metadata[id_col_name]
                    del metadata[id_col_name]

                    doc = Document(page_content=page_content, metadata=metadata, id=id)
                    docs.append(asdict(doc))

        return docs

    ##
    # Helper functions below
    ##
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(self, **kwargs):
        """
        Helper: exponetially backoff if the LLM's rate limit is exceeded.
        """
        traced_chat_completions_create_fn = mlflow.trace(
            self.llm_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )
        return traced_chat_completions_create_fn(**kwargs)

    def chat_completion(self, messages: List[ChatMessage]) -> ChatCompletionResponse:
        """
        Helper: Call the LLM configured via the ModelConfig using the OpenAI SDK
        """
        request = {
            "messages": messages,
            "temperature": self.maud_config.model.parameters.temperature,
            "max_tokens": self.maud_config.model.parameters.max_tokens,
            "tools": self.retriever_tool_spec,
        }

        return self.completions_with_backoff(
            model=self.maud_config.model.endpoint_name,
            **request,
        )

    @mlflow.trace(span_type="CHAIN")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        """
        Helper: Recursively calls the LLM w/ the tools in the prompt.  Either executes the tools and recalls the LLM or returns the LLM's generation.
        """
        messages = kwargs["messages"]
        del kwargs["messages"]
        i = 0
        while i < max_iter:
            with mlflow.start_span(name=f"iteration_{i}", span_type="CHAIN") as span:
                response = self.chat_completion(messages=messages)
                assistant_message = response.choices[0].message  # openai client
                tool_calls = assistant_message.tool_calls  # openai
                if tool_calls is None:
                    # the tool execution finished, and we have a generation
                    return response.to_dict()
                tool_messages = []
                for tool_call in tool_calls:  # TODO: should run in parallel
                    with mlflow.start_span(
                        name="execute_tool", span_type="TOOL"
                    ) as span:
                        function = tool_call.function
                        args = json.loads(function.arguments)
                        span.set_inputs(
                            {
                                "function_name": function.name,
                                "function_args_raw": function.arguments,
                                "function_args_loaded": args,
                            }
                        )
                        result = self.execute_function(
                            self.tool_functions[function.name], args
                        )
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }

                        tool_messages.append(tool_message)
                        span.set_outputs({"new_message": tool_message})
                assistant_message_dict = assistant_message.dict().copy()
                del assistant_message_dict["content"]
                del assistant_message_dict["function_call"]

                if "audio" in assistant_message_dict:
                    del assistant_message_dict["audio"]  # hack to make llama70b work

                messages = (
                    messages
                    + [
                        assistant_message_dict,
                    ]
                    + tool_messages
                )
                i += 1

        # TODO: Handle more gracefully
        raise "ERROR: max iter reached"

    def execute_function(self, tool, args):
        """
        Execute a tool and return the result as a JSON string
        """
        result = tool(**args)
        return json.dumps(result)


def convert_chat_messages_to_dict(messages: List[ChatMessage]):
    new_messages = []
    for message in messages:
        if type(message) == ChatMessage:
            # Remove any keys with None values
            new_messages.append(
                {k: v for k, v in asdict(message).items() if v is not None}
            )
        else:
            new_messages.append(message)
    return new_messages


# tell MLflow logging where to find the agent's code
mlflow.models.set_model(FunctionCallingAgent())
