from langgraph.graph import StateGraph, END

from app.agent.state import State
from app.agent.nodes.classification import classification_node
from app.agent.nodes.entity_extraction import entity_extraction_node
from app.agent.nodes.sentiment import sentiment_node
from app.agent.nodes.summarization import summarization_node
from app.agent.nodes.memory_retrieval import memory_retrieval_node
from app.agent.nodes.memory_save import memory_save_node


def route_after_classification(state: State) -> list[str]:
    """
    Routing function for the conditional edge after classification.

    Returns a list of nodes to execute next (fan-out):
    - News or Research: run entity_extraction AND sentiment in parallel
    - Blog or Other: run only sentiment (skip entity extraction)
    """
    category = state["classification"].lower().strip()
    if category in ["news", "research"]:
        return ["entity_extraction", "sentiment"]
    return ["sentiment"]


def build_graph() -> StateGraph:
    """
    Constructs and compiles the text analysis graph.

    The graph follows this flow:

        memory_retrieval
               |
        classification_node
               |
        [conditional edge]
          /              \
    [News/Research]    [Blog/Other]
         |                  |
      [fan-out]             |
      /      \              |
    entity  sentiment   sentiment
      \      /              |
      [fan-in]              |
           \               /
            [summarization]
                  |
            [memory_save]
                  |
                [END]
    """
    graph = StateGraph(State)

    # --- Register nodes ---
    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("classification_node", classification_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("memory_save", memory_save_node)

    # --- Entry point ---
    graph.set_entry_point("memory_retrieval")

    # --- Memory retrieval runs first, then classification ---
    graph.add_edge("memory_retrieval", "classification_node")

    # --- Conditional edge with fan-out ---
    graph.add_conditional_edges(
        "classification_node",
        route_after_classification,
        ["entity_extraction", "sentiment"]
    )

    # --- Fan-in: both parallel nodes point to summarization ---
    graph.add_edge("entity_extraction", "summarization")
    graph.add_edge("sentiment", "summarization")

    # --- Save memories then end ---
    graph.add_edge("summarization", "memory_save")
    graph.add_edge("memory_save", END)

    return graph.compile()


# Singleton: compile the graph once at module load time.
agent = build_graph()