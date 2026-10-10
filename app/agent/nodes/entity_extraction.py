from langchain_core.output_parsers import StrOutputParser

from app.agent.state import State
from app.prompts.entity_extraction import entity_extraction_prompt

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def entity_extraction_node(state: State) -> dict:
    chain = entity_extraction_prompt | llm | StrOutputParser()
    entities = chain.invoke({"text": state["text"]})
    return {"entities": entities.strip()}