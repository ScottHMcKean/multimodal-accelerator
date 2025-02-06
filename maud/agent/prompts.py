from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

prompt_template_no_history = """You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question."""

prompt_no_history = ChatPromptTemplate.from_messages(
[
    ("system", prompt_template_no_history),
    MessagesPlaceholder(variable_name="question")
]
)

prompt_template_with_history = """Based on conversation history below, rephrase the user's last question to capture context from the prior conversation that is necessary to answer the user's last question. The refrased question will be used to perform a semantic similarity search to find documents relevent to the user's last question. Do not answer the question. Simply return the refrased question without any explanation"""

prompt_with_history = ChatPromptTemplate.from_messages(
[
    ("system", prompt_template_with_history),
    MessagesPlaceholder(variable_name="messages")
]
)