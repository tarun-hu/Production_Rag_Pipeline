# 🔍 Enterprise RAG System

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, LangGraph, and Supabase. Upload PDFs, ask questions, and get grounded AI answers — with full security, caching, and observability.

---

## Architecture

```
User ──► Streamlit UI ──► FastAPI Backend
                              │
                    ┌─────────┴──────────┐
                    │  Input Security     │
                    │  L1 Regex           │
                    │  L4a JWT Auth       │
                    │  L4b Rate Limit     │
                    │  L5 Truncation      │
                    │  L2 llm-guard       │
                    │  L7a PII Redact     │
                    └─────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │  LangGraph Pipeline  │
                    │                      │
                    │  Answer Cache Check  │
                    │  HyDE Generation     │
                    │  Embed (+ cache)     │
                    │  Hybrid Retrieval    │
                    │  RRF Fusion (k=60)   │
                    │  Cross-Encoder Rerank│
                    │  CRAG + Tavily       │
                    │  XML Spotlight       │
                    │  LLM Generation      │
                    │  Self-RAG Reflect    │
                    │  Answer Cache Store  │
                    └─────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Output Security     │
                    │  L7b PII Redact      │
                    │  L9 Pydantic Valid.   │
                    └─────────┬──────────┘
                              ▼
                         User Response
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| API Framework | FastAPI |
| Auth | Supabase Auth + JWT (PyJWT) |
| Rate Limiting | `upstash-ratelimit` (SlidingWindow, 20 req/min) |
| Caching | Upstash Redis (Embedding: 7d TTL, Answer: 1h TTL) |
| Document Parsing | LlamaIndex (`PDFReader` + `SentenceSplitter`) |
| Vector DB | Supabase PostgreSQL + `pgvector` |
| Orchestration | LangGraph (11-node state machine) |
| LLM | DeepSeek V4 Pro via NVIDIA NIM API |
| Embeddings | `nvidia/nv-embedqa-e5-v5` (1024 dims) via NVIDIA NIM |
| Reranking | `flashrank` (CPU cross-encoder) |
| Web Fallback | Tavily API |
| Input Security | `llm-guard`, regex, tiktoken truncation |
| UI | Streamlit |

---

## Project Structure

```
Production_Rag_Pipeline/
├── main.py            # FastAPI server — routes, auth, security wiring
├── config.py          # Central configuration (all env vars + helpers)
├── auth.py            # Supabase JWT verification (JWKS)
├── rate_limiter.py    # Upstash sliding-window rate limiter
├── security.py        # Input/Output security pipeline (L1-L9)
├── cache.py           # Two-tier caching (embedding + RAG answer)
├── ingestion.py       # PDF parse → chunk → embed → store pipeline
├── vector_store.py    # Supabase pgvector CRUD operations
├── rag_graph.py       # LangGraph RAG pipeline (11 nodes)
├── ui.py              # Streamlit chat UI
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

---

## Setup

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd Production_Rag_Pipeline
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

| Variable | Source |
|----------|--------|
| `SUPABASE_URL` | Supabase project → Settings → API |
| `SUPABASE_ANON_KEY` | Supabase project → Settings → API |
| `SUPABASE_DB_URL` | Supabase project → Settings → Database → Connection string |
| `NVIDIA_API_KEY` | [NVIDIA NIM](https://build.nvidia.com/) |
| `NVIDIA_EMBEDDING_API_KEY` | NVIDIA NIM (can be a separate key for embeddings) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) (optional — enables web fallback) |
| `UPSTASH_REDIS_REST_URL` | [Upstash Console](https://console.upstash.com/) |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Console |

### 3. Run Supabase SQL Migration

Open your **Supabase SQL Editor** and execute:

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table (1024 dims for nv-embedqa-e5-v5)
CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content text,
    metadata jsonb,
    embedding vector(1024)
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS documents_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

-- Similarity search function
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1024),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### 4. Run the Application

**Terminal 1 — FastAPI backend:**
```bash
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run ui.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Sign up / Login** using the sidebar (Supabase Auth)
2. **Upload a PDF** via the sidebar file uploader
3. **Ask questions** about the PDF content in the chat
4. **View execution traces** — expand any response to see cache hits, source citations, and security details

---

## Pipeline Stages (Build Order)

| Stage | What It Does | File(s) |
|-------|-------------|---------|
| 0 | Project setup, dependencies, config | `config.py`, `requirements.txt` |
| 1 | Input security (regex, tiktoken, llm-guard, PII) | `security.py` |
| 2 | PDF ingestion (parse, chunk, embed, store) | `ingestion.py`, `vector_store.py` |
| 3 | LangGraph skeleton + PostgresSaver | `rag_graph.py` |
| 4 | Embedding cache (SHA-256, 7d TTL) | `cache.py` |
| 5 | RAG pipeline (HyDE, retrieval, RRF, rerank, CRAG) | `rag_graph.py` |
| 6 | LLM generation + Self-RAG reflection | `rag_graph.py` |
| 7 | RAG answer cache (1h TTL) | `cache.py`, `rag_graph.py` |
| 8 | Output security (PII redact, Pydantic validate) | `security.py`, `main.py` |
| 9 | Streamlit UI + end-to-end wiring | `ui.py`, `main.py` |

---

## Definition of Done

- [x] 401 on missing/invalid JWT
- [x] 429 on rate limit exceeded (21st request in a minute)
- [x] PDF upload → parse → chunk → embed → index in pgvector
- [x] Injection attempts blocked by input security pipeline
- [x] RAG answers grounded on uploaded PDF content
- [x] CRAG → Tavily web fallback when local results are weak
- [x] Cache hits on repeated queries (embedding + answer)
- [x] Pydantic validation with retry-on-failure
- [x] Full flow works through Streamlit UI

---

## Architecture Decisions

1. **Unified Supabase**: A single Supabase project handles auth, relational data, pgvector storage, and LangGraph checkpoints — eliminating the need for separate ChromaDB/Qdrant/SQLite instances.

2. **NVIDIA NIM API**: Using DeepSeek V4 Pro for generation and `nv-embedqa-e5-v5` for embeddings, both through NVIDIA's OpenAI-compatible API. Separate API keys are supported for LLM vs embedding calls.

3. **Two-tier cache**: Embedding cache (7d) catches repeated queries before any retrieval work. Answer cache (1h) skips the entire pipeline for exact-match questions. Both use SHA-256 normalized keys.

4. **Flashrank over API-based rerankers**: CPU-based cross-encoder reranking avoids additional API calls and latency, keeping the rerank step fast and free.

5. **Graceful degradation**: If llm-guard isn't fully installed, the security pipeline skips L2 scanning but still runs all other layers. If Tavily isn't configured, CRAG falls back to local results only.
