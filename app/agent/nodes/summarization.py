from langchain_core.output_parsers import StrOutputParser

from app.agent.state import State
from app.prompts.summarization import summarization_prompt

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def summarization_node(state: State) -> dict:
    memories = state.get("memories", [])
    memory_context = "\n".join([m.content for m in memories]) if memories else "None"

    chain = summarization_prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": state["text"], "memories": memory_context})
    return {"summary": summary.strip()}