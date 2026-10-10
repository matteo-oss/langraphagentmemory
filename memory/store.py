"""
store.py — Redis Memory Storage & Retrieval

Handles all interactions with Redis for long-term memory:
    1. Embedding generation using HuggingFace (all-MiniLM-L6-v2)
    2. Deduplication — prevents storing similar memories twice
    3. Storage — saves memories with vector embeddings to Redis
    4. Retrieval — finds relevant memories using semantic search

Why vector embeddings?
    Instead of exact keyword matching, we store each memory as a
    vector of numbers representing its meaning. When searching,
    we find memories that are semantically similar to the query —
    even if they use completely different words.

    Example:
        Stored: "User is always interested in AI stories"
        Query:  "What topics does this user care about?"
        → Found via semantic similarity, not keyword match
"""

import logging
import os
import traceback

from datetime import datetime
from typing import List, Optional, Union
import numpy as np

import ulid

from dotenv import load_dotenv
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.query.filter import Tag
from redisvl.schema.schema import IndexSchema
from sentence_transformers import SentenceTransformer

from memory.models import Memory, MemoryType, StoredMemory

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = Redis.from_url(REDIS_URL, decode_responses=False)

# ---------------------------------------------------------------------------
# HuggingFace Embeddings
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
# Lightweight, fast, and free — good enough for our use case.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIMS = 384


def embed(text: str) -> List[float]:
    """
    Convert a text string into a vector embedding.

    The embedding captures the semantic meaning of the text,
    allowing us to find similar memories even with different wording.

    Args:
        text: The text to embed.

    Returns:
        A list of 384 floats representing the text's meaning.
    """
    return embedding_model.encode(text).tolist()


# ---------------------------------------------------------------------------
# Redis Search Index
# ---------------------------------------------------------------------------

# The index schema defines the structure of our memory storage in Redis.
# Each memory is stored as a JSON document with the following fields:
#   - content: the memory text (searchable)
#   - memory_type: episodic or semantic (filterable tag)
#   - metadata: extra context as JSON string
#   - created_at: timestamp
#   - user_id: owner of the memory (filterable tag)
#   - memory_id: unique identifier (filterable tag)
#   - thread_id: source conversation (filterable tag)
#   - embedding: vector representation for semantic search

memory_schema = IndexSchema.from_dict({
    "index": {
        "name": "hn_agent_memories",
        "prefix": "memory",
        "key_separator": ":",
        "storage_type": "json",
    },
    "fields": [
        {"name": "content", "type": "text"},
        {"name": "memory_type", "type": "tag"},
        {"name": "metadata", "type": "text"},
        {"name": "created_at", "type": "text"},
        {"name": "user_id", "type": "tag"},
        {"name": "memory_id", "type": "tag"},
        {"name": "thread_id", "type": "tag"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": EMBEDDING_DIMS,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
    ],
})

# Create the index on startup — overwrite if it already exists.
try:
    memory_index = SearchIndex(
        schema=memory_schema,
        redis_client=redis_client,
        validate_on_load=True,
    )
    memory_index.create(overwrite=True)
    logger.info("Memory index ready")
except Exception as e:
    logger.error(f"Error creating memory index: {e}")
    raise

# ---------------------------------------------------------------------------
# Default user ID for memories not tied to a specific user
# ---------------------------------------------------------------------------

SYSTEM_USER_ID = "system"

# ---------------------------------------------------------------------------
# 1. Deduplication
# ---------------------------------------------------------------------------


def similar_memory_exists(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.15,
) -> bool:
    """
    Check if a semantically similar memory already exists in Redis.

    Before storing a new memory, we check if something similar is already
    stored. This prevents redundant memories like:
        "User likes AI stories"
        "User is interested in AI content"   ← too similar, skip

    Args:
        content: The memory text to check for duplicates.
        memory_type: EPISODIC or SEMANTIC.
        user_id: The user to scope the search to.
        thread_id: Optional thread to narrow the search.
        distance_threshold: How similar is "too similar" (0=identical, 1=opposite).
                            0.15 means 85%+ similarity → skip storage.

    Returns:
        True if a similar memory exists (skip storage), False otherwise.
    """
    content_embedding = embed(content)

    filters = (Tag("user_id") == user_id) & (Tag("memory_type") == memory_type)
    if thread_id:
        filters = filters & (Tag("thread_id") == thread_id)

    vector_query = VectorRangeQuery(
        vector=content_embedding,
        num_results=1,
        vector_field_name="embedding",
        filter_expression=filters,
        distance_threshold=distance_threshold,
        return_fields=["id"],
    )

    results = memory_index.query(vector_query)

    if results:
        logger.debug(f"Similar memory found: {results[0]['id']} — skipping storage")
        return True

    return False


# ---------------------------------------------------------------------------
# 2. Storage
# ---------------------------------------------------------------------------


def store_memory(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    metadata: Optional[str] = None,
) -> None:
    """
    Store a long-term memory in Redis with deduplication.

    Flow:
        1. Check if a similar memory already exists → skip if so
        2. Generate a vector embedding for the content
        3. Save the memory document to Redis with all metadata

    Args:
        content: The memory text to store.
                 Example: "User always asks for bullet-point summaries"
        memory_type: EPISODIC (user preference) or SEMANTIC (general knowledge).
        user_id: The user this memory belongs to.
        thread_id: The conversation thread this memory came from.
        metadata: Optional JSON string for extra context.
    """
    if metadata is None:
        metadata = "{}"

    logger.info(f"Preparing to store memory: {content}")

    # Deduplication check
    if similar_memory_exists(content, memory_type, user_id, thread_id):
        logger.info("Similar memory already exists — skipping")
        return

    embedding = embed(content)

    memory_data = {
        "user_id": user_id or SYSTEM_USER_ID,
        "content": content,
        "memory_type": memory_type.value,
        "metadata": metadata,
        "created_at": datetime.now().isoformat(),
        "embedding": np.array(embedding, dtype=np.float32),
        "memory_id": str(ulid.new()),
        "thread_id": thread_id or "",
    }

    try:
        memory_index.load([memory_data], id_field="memory_id")
        logger.info(f"Stored {memory_type.value} memory: {content}")
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 3. Retrieval
# ---------------------------------------------------------------------------


def retrieve_memories(
    query: str,
    memory_type: Union[Optional[MemoryType], List[MemoryType]] = None,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.3,
    limit: int = 5,
) -> List[StoredMemory]:
    """
    Retrieve memories from Redis using semantic similarity search.

    Instead of exact keyword matching, we find memories whose meaning
    is closest to the query — even if they use different words.

    Args:
        query: What we're looking for.
               Example: "What HN stories has this user analyzed before?"
        memory_type: Filter by EPISODIC, SEMANTIC, or both (pass a list).
                     None means search across all types.
        user_id: The user to retrieve memories for.
        thread_id: Optional — scope to a specific conversation.
        distance_threshold: Maximum semantic distance to include (0=identical, 1=opposite).
                            0.3 means we return memories with 70%+ similarity.
        limit: Maximum number of memories to return.

    Returns:
        A list of StoredMemory objects, ordered by similarity.
    """
    logger.debug(f"Retrieving memories for query: {query}")

    vector_query = VectorRangeQuery(
        vector=embed(query),
        return_fields=[
            "content",
            "memory_type",
            "metadata",
            "created_at",
            "memory_id",
            "thread_id",
            "user_id",
        ],
        num_results=limit,
        vector_field_name="embedding",
        dialect=2,
        distance_threshold=distance_threshold,
    )

    # Build filter string
    base_filters = [f"@user_id:{{{user_id or SYSTEM_USER_ID}}}"]

    if memory_type:
        if isinstance(memory_type, list):
            base_filters.append(f"@memory_type:{{{'|'.join(memory_type)}}}")
        else:
            base_filters.append(f"@memory_type:{{{memory_type.value}}}")

    if thread_id:
        base_filters.append(f"@thread_id:{{{thread_id}}}")

    vector_query.set_filter(" ".join(base_filters))

    results = memory_index.query(vector_query)

    memories = []
    for doc in results:
        try:
            memory = StoredMemory(
                id=doc["id"],
                memory_id=doc["memory_id"],
                user_id=doc["user_id"],
                thread_id=doc.get("thread_id") or None,
                memory_type=MemoryType(doc["memory_type"]),
                content=doc["content"],
                created_at=doc["created_at"],
                metadata=doc["metadata"],
            )
            memories.append(memory)
        except Exception as e:
            logger.error(f"Error parsing memory doc: {e}")
            continue

    logger.debug(f"Found {len(memories)} memories for query: {query}")
    return memories
