from langchain_core.prompts import ChatPromptTemplate

summarization_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in one concise sentence.\n\n"
    "User preferences from memory:\n{memories}\n\n"
    "Text: {text}\n\nSummary:"
)