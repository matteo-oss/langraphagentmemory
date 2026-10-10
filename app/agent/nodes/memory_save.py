"""
memory_save.py - Memory Save Node

Runs at the END of the graph, after summarization.
"""

import logging

from app.agent.state import State
from memory.memory_manager import save_conversation_memories

logger = logging.getLogger(__name__)


def memory_save_node(state: State) -> dict:
    """
    Extract and store memories from the current conversation.

    Reconstructs a conversation from state fields and runs the
    full extraction pipeline -> store_memory().

    Returns:
        Empty dict - this node only writes to Redis.
    """
    user_id = state.get("user_id")

    if not user_id:
        logger.warning("No user_id in state - skipping memory save")
        return {}

    messages = [
        {"role": "user", "content": state["text"]},
        {
            "role": "assistant",
            "content": (
                f"Classification: {state.get('classification', 'N/A')}\n"
                f"Sentiment: {state.get('sentiment', 'N/A')}\n"
                f"Entities: {state.get('entities', 'N/A')}\n"
                f"Summary: {state.get('summary', 'N/A')}"
            ),
        },
    ]

    count = save_conversation_memories(
        messages=messages,
        user_id=user_id,
        thread_id=state.get("thread_id"),
    )

    logger.info(f"Saved {count} memories for user {user_id}")
    return {}
