from langchain_core.output_parsers import StrOutputParser

from app.agent.state import State
from app.prompts.sentiment import sentiment_prompt

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def sentiment_node(state: State) -> dict:
    """
    Analyzes the emotional tone of the input text.

    Uses the sentiment prompt to classify the text as
    Positive, Negative, or Neutral.

    Args:
        state: The current graph state containing the input text.

    Returns:
        A dict with the 'sentiment' field populated.
    """
    chain = sentiment_prompt | llm | StrOutputParser()
    sentiment = chain.invoke({"text": state["text"]})
    return {"sentiment": sentiment.strip()}