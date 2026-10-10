from typing import Optional, TypedDict
from memory.models import StoredMemory


class State(TypedDict):
    """
    Represents the shared state passed between all nodes in the graph.

    Fields are populated progressively as the graph executes:
    - text: provided at the start (required)
    - user_id: identifies the user for memory retrieval/storage
    - thread_id: identifies the conversation thread
    - memories: relevant memories retrieved at the start of the conversation
    - classification: filled by classification_node
    - entities: filled by entity_extraction_node (skipped for Blog/Other)
    - sentiment: filled by sentiment_node (always runs)
    - summary: filled by summarization_node
    """
    text: str
    user_id: str
    thread_id: Optional[str]
    memories: Optional[list[StoredMemory]]
    classification: Optional[str]
    entities: Optional[str]
    sentiment: Optional[str]
    summary: Optional[str]