from langchain_core.output_parsers import StrOutputParser

from app.agent.state import State
from app.prompts.classification import classification_prompt

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def classification_node(state: State) -> dict:
    chain = classification_prompt | llm | StrOutputParser()
    classification = chain.invoke({"text": state["text"]})
    return {"classification": classification.strip()}