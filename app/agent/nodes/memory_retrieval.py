"""
memory_retrieval.py — Memory Retrieval Node

Runs at the START of the graph, before any analysis.
Fetches relevant memories for the current user from Redis
and injects them into the state for all downstream nodes.
"""

import logging

from app.agent.state import State
from memory.memory_manager import get_relevant_memories

logger = logging.getLogger(__name__)


def memory_retrieval_node(state: State) -> dict:
    """
    Retrieve relevant long-term memories for the current user.

    Queries Redis using semantic similarity against the input text.
    Results are stored in state["memories"] so all downstream nodes
    (summarization, etc.) can use them without querying Redis again.

    Args:
        state: The current graph state. Requires:
               - state["text"]: used as the search query
               - state["user_id"]: scopes the search to this user

    Returns:
        dict with "memories" key containing a list of StoredMemory objects.
        Returns empty list if user_id is missing or no memories found.
    """
    user_id = state.get("user_id")

    if not user_id:
        logger.warning("No user_id in state — skipping memory retrieval")
        return {"memories": []}

    text = state["text"]

    memories = get_relevant_memories(query=text, user_id=user_id)

    logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
    return {"memories": memories}