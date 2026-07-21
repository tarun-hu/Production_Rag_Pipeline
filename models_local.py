"""
models_local.py — Local Model Registry.

Loads and caches the SentenceTransformer (embedding) and CrossEncoder (reranker)
models locally on CUDA (GPU) if available, with CPU fallback.
"""

import logging
import torch

from config import EMBEDDING_MODEL

logger = logging.getLogger("models_local")

# ── Global Cached Instances ──────────────────────────────────────────
_embedding_model = None
_reranker_model = None


def get_embedding_model():
    """
    Lazily initialize and return the local embedding SentenceTransformer model.
    Runs on GPU (CUDA) if available.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing local embedding model '{EMBEDDING_MODEL}' on '{device}'...")
            
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            logger.info("Local embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            raise
    return _embedding_model


def get_reranker_model():
    """
    Lazily initialize and return the local reranker CrossEncoder model.
    Runs on GPU (CUDA) if available.
    """
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing local CrossEncoder reranker 'BAAI/bge-reranker-base' on '{device}'...")
            
            _reranker_model = CrossEncoder("BAAI/bge-reranker-base", device=device)
            logger.info("Local reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local reranker model: {e}")
            raise
    return _reranker_model
