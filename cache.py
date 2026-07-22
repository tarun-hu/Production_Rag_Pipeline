"""
cache.py — Two-tier caching layer using Upstash Redis.

Tier 1: Embedding Cache  (7-day TTL)  — caches query embeddings
Tier 2: RAG Answer Cache (1-hour TTL) — caches full pipeline answers
"""

import hashlib
import json
import logging
from typing import Optional

from upstash_redis import Redis

from config import (
    UPSTASH_REDIS_REST_URL,
    UPSTASH_REDIS_REST_TOKEN,
    EMBEDDING_CACHE_TTL,
    RAG_ANSWER_CACHE_TTL,
)

logger = logging.getLogger("cache")

# ── Upstash Redis client ─────────────────────────────────────────────
redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)


# =====================================================================
# Cache Key Helper
# =====================================================================

def cache_key(prefix: str, text: str, user_id: Optional[str] = None) -> str:
    """
    Generate a deterministic cache key using SHA-256.

    Normalizes text (.strip().lower()) before hashing.
    If user_id is provided, includes it in the key to isolate data per user.
    """
    normalized = text.strip().lower()
    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if user_id:
        return f"{prefix}:{user_id}:{text_hash}"
    return f"{prefix}:{text_hash}"


# =====================================================================
# Embedding Cache (Tier 1)
# =====================================================================

def get_cached_embedding(text: str) -> Optional[list[float]]:
    """Check the embedding cache. Returns the vector or None on miss."""
    key = cache_key("emb", text)
    try:
        cached = redis.get(key)
        if cached:
            logger.info(f"EMBEDDING CACHE HIT: {key[:30]}...")
            # Upstash returns string, parse it back to list
            if isinstance(cached, str):
                return json.loads(cached)
            return cached
    except Exception as e:
        logger.error(f"Embedding cache read error: {e}")
    logger.info(f"EMBEDDING CACHE MISS: {key[:30]}...")
    return None


def set_cached_embedding(text: str, embedding: list[float]) -> None:
    """Store an embedding in the cache with 7-day TTL."""
    key = cache_key("emb", text)
    try:
        redis.set(key, json.dumps(embedding), ex=EMBEDDING_CACHE_TTL)
        logger.info(f"EMBEDDING CACHED: {key[:30]}... (TTL={EMBEDDING_CACHE_TTL}s)")
    except Exception as e:
        logger.error(f"Embedding cache write error: {e}")


# =====================================================================
# RAG Answer Cache (Tier 2)
# =====================================================================

def get_cached_answer(query: str, user_id: str) -> Optional[str]:
    """Check the RAG answer cache isolated by user_id."""
    key = cache_key("rag", query, user_id=user_id)
    try:
        cached = redis.get(key)
        if cached:
            logger.info(f"RAG ANSWER CACHE HIT: {key[:35]}...")
            return cached if isinstance(cached, str) else json.loads(cached)
    except Exception as e:
        logger.error(f"Answer cache read error: {e}")
    logger.info(f"RAG ANSWER CACHE MISS: {key[:35]}...")
    return None


def set_cached_answer(query: str, answer: str, user_id: str) -> None:
    """Store a RAG answer in the cache isolated by user_id."""
    key = cache_key("rag", query, user_id=user_id)
    try:
        redis.set(key, answer, ex=RAG_ANSWER_CACHE_TTL)
        logger.info(f"RAG ANSWER CACHED: {key[:35]}... (TTL={RAG_ANSWER_CACHE_TTL}s)")
    except Exception as e:
        logger.error(f"Answer cache write error: {e}")


def clear_user_cache(user_id: str) -> int:
    """Clear all answer cache entries for a specific user from Upstash Redis."""
    try:
        cursor = 0
        keys_to_delete = []
        pattern = f"rag:{user_id}:*"
        cursor, keys = redis.scan(cursor=cursor, match=pattern, count=2000)
        keys_to_delete.extend(keys)
        while cursor != 0:
            cursor, keys = redis.scan(cursor=cursor, match=pattern, count=2000)
            keys_to_delete.extend(keys)
            
        if keys_to_delete:
            for k in keys_to_delete:
                redis.delete(k)
            logger.info(f"Cleared {len(keys_to_delete)} cache keys for user '{user_id}'")
            return len(keys_to_delete)
    except Exception as e:
        logger.error(f"Failed to clear cache for user '{user_id}': {e}")
    return 0


def flush_all_redis_cache() -> bool:
    """Flush the entire Upstash Redis database."""
    try:
        redis.flushdb()
        logger.info("Flushed entire Upstash Redis cache database.")
        return True
    except Exception as e:
        logger.error(f"Failed to flush Redis: {e}")
        return False


# =====================================================================
# Cached Embedding Call Wrapper
# =====================================================================

def get_embedding(text: str) -> list[float]:
    """
    Wrapper that checks the embedding cache before computing the vector locally.

    On cache miss, computes the embedding using BAAI/bge-small-en-v1.5 on GPU/CPU.
    """
    # Check cache first
    cached = get_cached_embedding(text)
    if cached is not None:
        return cached

    # Cache miss → compute local embedding
    import models_local
    
    model = models_local.get_embedding_model()
    # Add BGE recommended query instruction prefix for search queries
    query_text = f"Represent this sentence for searching relevant passages: {text}"
    
    # Generate embedding and convert to list of floats
    embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    # Store in cache
    set_cached_embedding(text, embedding)
    return embedding
