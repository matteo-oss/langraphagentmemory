from langchain_core.prompts import ChatPromptTemplate

classification_prompt = ChatPromptTemplate.from_template(
    "Classify the following text into exactly one of these categories: "
    "News, Blog, Research, or Other.\n\n"
    "Text: {text}\n\n"
    "Respond with the category name only."
)