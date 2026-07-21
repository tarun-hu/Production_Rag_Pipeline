"""
vector_store.py — Local Qdrant document store interface with user-level isolation.

Manages collection initialization, document insertion, similarity search,
and document retrieval filtered by Supabase user_id.
"""

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from config import QDRANT_PATH, EMBEDDING_DIMENSIONS

logger = logging.getLogger("vector_store")

# ── Lazy initialization of Qdrant client ─────────────────────────────
_client = None
COLLECTION_NAME = "documents"


def _get_client() -> QdrantClient:
    """Lazily initialize and return the persistent Qdrant client."""
    global _client
    if _client is None:
        logger.info(f"Initializing local Qdrant client at path: {QDRANT_PATH}")
        _client = QdrantClient(path=QDRANT_PATH)
        
        # Ensure the documents collection exists on start
        if not _client.collection_exists(collection_name=COLLECTION_NAME):
            logger.info(f"Creating collection '{COLLECTION_NAME}' with {EMBEDDING_DIMENSIONS} dimensions...")
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )
    return _client


# =====================================================================
# Write Operations
# =====================================================================

def add_documents(chunks: list[dict], user_id: str) -> int:
    """
    Insert document chunks into the local Qdrant database, isolated by user_id.

    Each chunk dict must contain:
      - content  (str):        the raw text of the chunk
      - embedding (list[float]): the vector embedding
      - metadata (dict):       source info (filename, page_number, etc.)

    Returns the number of chunks inserted.
    """
    if not chunks:
        return 0

    client = _get_client()
    points = []
    
    for chunk in chunks:
        point_id = str(uuid.uuid4())
        content = chunk["content"]
        meta = chunk.get("metadata", {})
        embedding = chunk["embedding"]
        
        # Ensure embedding dimensions match Qdrant configuration
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding size mismatch. Expected {EMBEDDING_DIMENSIONS}, got {len(embedding)}"
            )

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": content,
                    "metadata": meta,
                    "user_id": user_id,  # Track ownership for security isolation
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )
    logger.info(f"Upserted {len(points)} document chunks into Qdrant collection '{COLLECTION_NAME}' for user '{user_id}'")
    return len(points)


# =====================================================================
# Search Operations
# =====================================================================

def search(query_embedding: list[float], user_id: str, k: int = 5, threshold: float = 0.0) -> list[dict]:
    """
    Perform vector similarity search, strictly isolated to the user's documents.

    Returns a list of dicts: [{id, content, metadata, similarity}, ...]
    """
    client = _get_client()

    # Apply strict user filtering
    user_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            )
        ]
    )

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=user_filter,
        limit=k,
        score_threshold=threshold if threshold > 0 else None,
    )

    results = []
    for hit in search_result.points:
        results.append({
            "id": str(hit.id),
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata", {}),
            "similarity": float(hit.score),
        })

    logger.info(f"Qdrant vector search returned {len(results)} results (k={k}) for user '{user_id}'")
    return results


# =====================================================================
# Utility: Fetch all documents (for BM25 sparse search)
# =====================================================================

def get_all_documents(user_id: str) -> list[dict]:
    """
    Retrieve all document chunks belonging to a specific user.

    Used to build the BM25 index isolated by user.
    Returns: [{id, content, metadata}, ...]
    """
    client = _get_client()

    # Filter only this user's records
    user_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            )
        ]
    )

    # Use scroll to retrieve all records matching the filter
    scroll_result, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=user_filter,
        limit=10000,  # Arbitrary limit (high enough for typical document sizes)
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for point in scroll_result:
        results.append({
            "id": str(point.id),
            "content": point.payload.get("content", ""),
            "metadata": point.payload.get("metadata", {}),
        })

    logger.info(f"Retrieved {len(results)} documents for BM25 index for user '{user_id}'")
    return results


def get_document_count(user_id: Optional[str] = None) -> int:
    """
    Return the number of documents in the collection.
    If user_id is provided, counts only that user's documents.
    """
    client = _get_client()
    
    if user_id:
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )
        count_res = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=user_filter,
        )
    else:
        count_res = client.count(
            collection_name=COLLECTION_NAME,
        )
        
    return count_res.count
