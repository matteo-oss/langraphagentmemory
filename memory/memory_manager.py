"""
memory_manager.py — Single Interface for the Memory System

The agent should only interact with this file.
Orchestrates between extractor.py and store.py.

Two operations the agent needs:
    1. save_conversation_memories() — called at the END of a conversation
    2. get_relevant_memories()      — called at the START of a conversation
"""

import logging

from memory.models import Memory
from memory.extractor import extract_memories
from memory.store import store_memory, retrieve_memories

logger = logging.getLogger(__name__)


def save_conversation_memories(
    messages: list[dict],
    user_id: str,
    thread_id: str | None = None,
) -> int:
    """
    Extract and store memories from a completed conversation.

    Called at the END of a conversation. Runs the full pipeline:
        messages → extract_memories() → store_memory() for each

    Args:
        messages: The full conversation history.
        user_id: The user this conversation belongs to.
        thread_id: Optional thread identifier for tracing memories back
                   to their source conversation.

    Returns:
        Number of memories successfully stored.

    Usage:
        count = save_conversation_memories(
            messages=state["messages"],
            user_id="user_123",
            thread_id="thread_456",
        )
        logger.info(f"Saved {count} memories")
    """
    memories: list[Memory] = extract_memories(messages)

    if not memories:
        logger.info("No memories extracted — nothing to store")
        return 0

    stored = 0
    for memory in memories:
        store_memory(
            content=memory.content,
            memory_type=memory.memory_type,
            user_id=user_id,
            thread_id=thread_id,
        )
        stored += 1

    logger.info(f"Saved {stored}/{len(memories)} memories for user {user_id}")
    return stored


def get_relevant_memories(
    query: str,
    user_id: str,
    limit: int = 5,
) -> list[Memory]:
    """
    Retrieve memories relevant to the current query.

    Called at the START of a conversation or before the agent responds.
    Returns memories the agent should be aware of when answering.

    Args:
        query: What the user is asking about — used for semantic search.
        user_id: The user to retrieve memories for.
        limit: Maximum number of memories to return.

    Returns:
        List of StoredMemory objects, ordered by relevance.

    Usage:
        memories = get_relevant_memories(
            query=user_message,
            user_id="user_123",
        )
        context = "\n".join([m.content for m in memories])
    """
    memories = retrieve_memories(
        query=query,
        user_id=user_id,
        limit=limit,
    )

    logger.info(f"Retrieved {len(memories)} relevant memories for user {user_id}")
    return memories