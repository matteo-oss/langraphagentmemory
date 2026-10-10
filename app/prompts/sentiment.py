from langchain_core.prompts import ChatPromptTemplate


sentiment_prompt = ChatPromptTemplate.from_template(
    "Analyze the sentiment of the following text.\n"
    "Respond with exactly one word: Positive, Negative, or Neutral.\n\n"
    "Text: {text}\n\n"
    "Sentiment:"
)