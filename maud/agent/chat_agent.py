from random import randint
import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (ChatMessage, 
                              ChatCompletionResponse, 
                              ChatChoice, 
                              ChatChoiceDelta,
                              ChatChunkChoice,
                              ChatCompletionChunk)
from graph import load_graph


class GraphChatModel(ChatModel):
  """
  A mlflow ChatModel that invokes or streams a LangGraph
  workflow. 

  The LangGraph output dictionary is parsed into mlflow chat model outputs
  that vary depending on whether the graph is invoked or streamed.

  MLflow autologging for Langchain must be explicitly turned on. It is not on
  by default as is the case when logging a langchain model flavor.
  """
  def __init__(self):
    self.app = load_graph()
    mlflow.langchain.autolog()


  def format_chat_response(self, answer, message_history=None, stream=False):
    """
    Reformat the LangGraph dictionary output into mlflow chat model types.
    Streaming output requires the ChatCompletionChunk type; batch (invoke)
    output requires the ChatCompletionResponse type.

    The models answer to the users question is returned. The messages history, 
    if it exists, is returned as a custom output within the chat message type.
    """
    if stream:
      chat_completion_response = ChatCompletionChunk(
                                  choices=[ChatChunkChoice(delta=ChatChoiceDelta(
                                    role = 'assistant',
                                    content=answer
                                  )
                                  )]
                              ) 

    else:
      chat_completion_response = ChatCompletionResponse(
                                  choices=[ChatChoice(message=ChatMessage(role="assistant", 
                                                      content=answer))]
                              )

    if message_history:
      chat_completion_response.custom_outputs = {"message_history": message_history}

    return chat_completion_response
  

  def predict_stream(self, context, messages, params=None):
    """
    NOTE: This method is not supported by Databricks model serving yet. 
    Stream the application on the input messages. Selectively choose the output 
    from graph events (node executions) to return to the user. This is necessary 
    when the model is executed using the 'stream' method rather than the 'invoke' 
    method. Each event in the stream represent the output of a LangGraph node. 
    The event is a dictionary where the key is the node's name and the value is the 
    nodes output dictionary, which conforms to the graph's state dictionary.

    The output from nodes that invoke the model are return to the user. The
    last event returned from the stream will also include the message history
    as a custom output within the mlflow chat message type.
    """
    messages = {"messages": [message.to_dict() for message in messages]}
    for event in self.app.stream(messages):

      if 'generation_with_history' in event:
        rewritten_question = event['generation_with_history']['generated_question'][-1]['content']
        rewritten_question_with_context = f"""To incorporate context from our conversation, I've rewritten your question as:
       
        '{rewritten_question}'
        """
        yield self.format_chat_response(rewritten_question_with_context, stream=True)
   

      if 'generation_no_history' in event:
          message_history = event['generation_no_history']['messages']
          answer = message_history[-1]['content']
          yield self.format_chat_response(answer, message_history=message_history, stream=True)


  def predict(self, context, messages, params=None):
    """
    Apply the invoke(batch) method to the LangGraph application. The entire graph will
    execute and the completed state dictionary will be returned by the model.

    Receives a list of messages in mlflow ChatMessage format. Convert the messages to a list
    of dictionaries before passing them to the application.

    Elements from the dictionary are retrieved and reformatted into the expected MLflow 
    chat model type and are then returned.
    """
    messages = {"messages": [message.to_dict() for message in messages]}
    generation = self.app.invoke(messages)
    message_history = generation['messages']
    answer = message_history[-1]['content']
    return self.format_chat_response(answer, message_history=message_history)

mlflow.models.set_model(GraphChatModel())
