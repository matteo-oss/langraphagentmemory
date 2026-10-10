"""
models.py — Memory Data Models

Defines the data structures used throughout the memory system.
These models handle the complete memory lifecycle:
from creation (Memory) to storage and retrieval (StoredMemory).

Three core components:
    - MemoryType: Categorizes memories into episodic or semantic
    - Memory: Represents a single memory before it's stored
    - StoredMemory: Represents a memory after it's been persisted to Redis
"""

import ulid

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """
    Defines the type of a long-term memory.

    EPISODIC: User-specific preferences and experiences.
              Stored and retrieved per user.
              Examples:
                - "User is always interested in AI-related stories"
                - "User prefers long summaries"

    SEMANTIC: General knowledge accumulated over time.
              Shared across users and sessions.
              Examples:
                - "Story 43291847 was already analyzed - positive sentiment"
                - "Stories about LLMs tend to have high engagement on HN"
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class Memory(BaseModel):
    """
    Represents a single memory before it is persisted to Redis.

    This is the input model — created by the LLM when it decides
    something is worth remembering.

    Attributes:
        content: The actual memory text to store.
                 Example: "User always asks for bullet-point summaries"
        memory_type: Whether this is a user preference (EPISODIC)
                     or general knowledge (SEMANTIC).
        metadata: Optional JSON string for extra context.
                  Example: '{"source": "user_request", "confidence": "high"}'
    """

    content: str
    memory_type: MemoryType
    metadata: str = "{}"


class StoredMemory(Memory):
    """
    Represents a memory that has been persisted to Redis.

    Extends Memory with storage metadata added at write time.
    This is the output model — returned when retrieving memories.

    Attributes:
        id: The Redis key where this memory is stored (e.g. "memory:01J...")
        memory_id: A unique ULID identifier for this memory entry.
        created_at: Timestamp of when the memory was stored.
        user_id: The user this memory belongs to.
                 None for semantic memories shared across users.
        thread_id: The conversation thread this memory originated from.
                   Useful for tracing a memory back to its source conversation.
        memory_type: Inherited from Memory — EPISODIC or SEMANTIC.
    """

    id: str
    memory_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None