"""
extractor.py — LLM-based Memory Extraction

Analyzes a conversation and extracts memories worth storing long-term.

Flow:
    1. Receive conversation history (list of messages)
    2. Split into chunks if conversation is too long
    3. Send each chunk to LLM with structured output (Pydantic)
    4. Merge results from all chunks
    5. Return list of Memory objects ready to store
"""

import logging

from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from memory.models import Memory, MemoryType

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Maximum number of messages to send to the LLM in a single chunk.
# Conversations longer than this will be split and processed separately.
MAX_MESSAGES_PER_CHUNK = 5

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_base_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


class ExtractedMemory(BaseModel):
    """A single memory extracted from the conversation."""
    content: str
    type: Literal["episodic", "semantic"]


class ExtractionResult(BaseModel):
    """The full structured output returned by the LLM."""
    memories: list[ExtractedMemory]


# Bind structured output to LLM — guarantees valid Pydantic object back,
# no need for json.loads() or manual error handling on the response.
llm = _base_llm.with_structured_output(ExtractionResult)

# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
You are a memory extraction assistant. Analyze the conversation below 
and extract ONLY the most important information worth remembering about the user.

Categories to focus on:
1. TOPIC INTERESTS - Core subjects (e.g., AI, cybersecurity, startups).
2. FORMAT PREFERENCES - How they prefer to receive data (e.g., "likes bullet points").
3. ANALYZED STORIES - Specific HN story IDs or titles they discussed.
4. RECURRING PATTERNS - Behavior that repeats across messages.

Strict Rules:
- LIMIT OUTPUT: Extract a maximum of 5-7 high-quality memories per chunk.
- QUALITY OVER QUANTITY: Do not list every single fact; only what defines the user's profile.
- Specificity: Use "User is interested in GrapheneOS" instead of "User likes tech".
- Categorization: 
    - episodic: Personal preferences, history, and behaviors.
    - semantic: Facts or general knowledge mentioned that are worth keeping.

Conversation:
{conversation}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_conversation(messages: list[dict]) -> str:
    """
    Convert a list of message dicts into a readable string for the prompt.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        Formatted string like:
            user: bring me AI articles
            assistant: Here are the top AI stories...
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def chunk_messages(messages: list[dict], chunk_size: int = MAX_MESSAGES_PER_CHUNK) -> list[list[dict]]:
    """
    Split a long conversation into smaller chunks.

    Why: LLMs have a context window limit. Sending 100 messages at once
    may exceed it, or produce lower quality extraction. We split into
    chunks so each one is processed cleanly.

    Args:
        messages: The full conversation history.
        chunk_size: Max messages per chunk.

    Returns:
        List of message chunks.

    Example:
        40 messages with chunk_size=20 -> [[msg0..msg19], [msg20..msg39]]
    """
    if len(messages) <= chunk_size:
        return [messages]

    chunks = []
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i: i + chunk_size]
        chunks.append(chunk)

    logger.info(f"Split {len(messages)} messages into {len(chunks)} chunks")
    return chunks


def extract_from_chunk(messages: list[dict]) -> list[Memory]:
    """
    Run extraction on a single chunk of messages.

    Args:
        messages: A chunk of conversation messages.

    Returns:
        List of Memory objects extracted from this chunk.
        Note: user_id and thread_id are not set here — pass them
        separately when calling store_memory().
    """
    conversation_text = format_conversation(messages)
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    try:
        result: ExtractionResult = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return []

    memories = []
    for item in result.memories:
        memory = Memory(
            content=item.content,
            memory_type=MemoryType(item.type),
        )
        memories.append(memory)

    return memories


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_memories(messages: list[dict]) -> list[Memory]:
    """
    Extract memories from a conversation using the LLM.

    Handles long conversations by splitting into chunks and merging results.
    user_id and thread_id are intentionally not set here — pass them
    separately when calling store_memory() so each module stays focused.

    Args:
        messages: The full conversation history to analyze.

    Returns:
        List of Memory objects ready to be stored via store_memory().

    Usage:
        memories = extract_memories(messages)
        for memory in memories:
            store_memory(
                content=memory.content,
                memory_type=memory.memory_type,
                user_id=user_id,
                thread_id=thread_id,
            )
    """
    if not messages:
        logger.info("No messages to extract from")
        return []

    logger.info(f"Extracting memories from {len(messages)} messages")

    chunks = chunk_messages(messages)

    all_memories: list[Memory] = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} messages)")
        chunk_memories = extract_from_chunk(chunk)
        all_memories.extend(chunk_memories)

    logger.info(f"Extracted {len(all_memories)} memories total")
    return all_memories
