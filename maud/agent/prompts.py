from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the following question preceded by So, yeah."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

context_prompt = """You are a trusted assistant that helps answer questions based only on the provided context. Here is some context which might or might not help you answer: {context}.  If the context is not helpful, you should say you do not know, and summarize the context in one sentence."""

context_template = ChatPromptTemplate.from_messages(
    [("system", context_prompt), MessagesPlaceholder(variable_name="question")]
)

rephrase_prompt = """Based on conversation history below, rephrase the user's last question to capture context from the prior conversation that is necessary to answer the user's last question. The refrased question will be used to perform a semantic similarity search to find documents relevent to the user's last question. Do not answer the question. Simply return the refrased question without any explanation"""

rephrase_template = ChatPromptTemplate.from_messages(
    [("system", rephrase_prompt), MessagesPlaceholder(variable_name="messages")]
)
