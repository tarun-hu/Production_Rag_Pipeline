"""
config.py — Central configuration for the Enterprise RAG System.
Loads all secrets and tunables from .env via os.environ.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Supabase ─────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "")  # postgresql://... (uses pooler on IPv4 networks)
JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

# ── NVIDIA NIM API (OpenAI-compatible) ───────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# ── LLM Provider (via NVIDIA NIM) ───────────────────────────────────
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/deepseek-v4-pro")

# ── Local Embedding Provider ─────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "384"))

# ── Local Qdrant Storage ─────────────────────────────────────────────
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant_db")

# ── Tavily (Web fallback) ────────────────────────────────────────────
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# ── Upstash Redis ────────────────────────────────────────────────────
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

# ── RAG Pipeline Thresholds ──────────────────────────────────────────
CRAG_RELEVANCE_THRESHOLD = float(os.environ.get("CRAG_RELEVANCE_THRESHOLD", "0.7"))
SELF_RAG_QUALITY_THRESHOLD = float(os.environ.get("SELF_RAG_QUALITY_THRESHOLD", "0.8"))
SELF_RAG_MAX_RETRIES = int(os.environ.get("SELF_RAG_MAX_RETRIES", "2"))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "15"))
HYDE_NUM_HYPOTHETICALS = int(os.environ.get("HYDE_NUM_HYPOTHETICALS", "3"))

# ── Cache TTLs (seconds) ─────────────────────────────────────────────
EMBEDDING_CACHE_TTL = int(os.environ.get("EMBEDDING_CACHE_TTL", str(7 * 24 * 3600)))  # 7 days
RAG_ANSWER_CACHE_TTL = int(os.environ.get("RAG_ANSWER_CACHE_TTL", str(7 * 24 * 3600))) # 7 days

# ── Input Security ───────────────────────────────────────────────────
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "256"))

# ── Ingestion ────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# ── Rate Limiting ────────────────────────────────────────────────────
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "5"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))


# ── Helper: get a shared OpenAI-compatible client for NVIDIA ─────────
def get_llm_client():
    """Return an OpenAI client configured for the NVIDIA NIM API."""
    from openai import OpenAI
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
