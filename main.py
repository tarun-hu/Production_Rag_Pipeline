"""
main.py — FastAPI application for the Enterprise RAG System.

Wires together:
  - Supabase JWT Authentication
  - Upstash Rate Limiting
  - Input Security Pipeline
  - PDF Ingestion (LlamaIndex → Supabase pgvector)
  - LangGraph RAG Pipeline
  - Output Security Pipeline
  - Pydantic Response Validation
"""

import logging
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth import get_current_user
from rate_limiter import rate_limit_dependency
from security import (
    QueryRequest,
    run_input_security_pipeline,
    run_output_security_pipeline,
)
from config import MAX_INPUT_TOKENS, SUPABASE_DB_URL

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-14s │ %(levelname)-5s │ %(message)s",
)
logger = logging.getLogger("main")

# ── Global Checkpointer & Connection Pool ────────────────────────────
checkpointer = None
pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, pool
    if SUPABASE_DB_URL:
        try:
            from psycopg_pool import ConnectionPool
            from langgraph.checkpoint.postgres import PostgresSaver

            logger.info("Initializing PostgresSaver checkpointer pool...")
            # Initialize connection pool. We use a max_size of 10 to be gentle on Supabase connection limits.
            # We set autocommit=True so CREATE INDEX CONCURRENTLY during checkpointer setup runs successfully.
            pool = ConnectionPool(conninfo=SUPABASE_DB_URL, max_size=10, kwargs={"autocommit": True})
            
            # Initialize PostgresSaver checkpointer
            checkpointer = PostgresSaver(pool)
            
            # Create checkpoints tables in the database if they do not exist
            checkpointer.setup()
            logger.info("PostgresSaver checkpointer initialized and tables set up successfully.")
        except Exception as e:
            logger.error(f"PostgresSaver setup failed: {e}")
            checkpointer = None
    yield
    # Shutdown connection pool on app exit
    if pool:
        logger.info("Closing connection pool...")
        pool.close()

# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG System",
    description="Production-grade RAG pipeline with security, caching, and LangGraph orchestration.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again later."},
    )


# =====================================================================
# Response Schemas
# =====================================================================

class RAGResponse(BaseModel):
    """Validated response schema for RAG queries."""
    query: str
    answer: str
    cache_hit: bool
    sources: list[dict] = []
    security_details: dict = {}


class IngestionResponse(BaseModel):
    """Response schema for PDF uploads."""
    status: str
    filename: str
    pages: int
    chunks: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    documents_indexed: int


# =====================================================================
# Public Routes
# =====================================================================

@app.get("/", response_model=HealthResponse)
def health_check():
    """Public health check endpoint."""
    try:
        import vector_store
        count = vector_store.get_document_count()
    except Exception:
        count = 0
    return HealthResponse(
        status="healthy",
        service="Enterprise RAG System",
        documents_indexed=count,
    )


# =====================================================================
# Auth Routes (proxy to Supabase)
# =====================================================================

class AuthRequest(BaseModel):
    email: str
    password: str


@app.post("/signup")
def signup(payload: AuthRequest):
    """Proxy signup to Supabase Auth."""
    import requests
    from config import SUPABASE_URL, SUPABASE_ANON_KEY

    response = requests.post(
        f"{SUPABASE_URL}/auth/v1/signup",
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json",
        },
        json={"email": payload.email, "password": payload.password},
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()


@app.post("/login")
def login(payload: AuthRequest):
    """Proxy login to Supabase Auth, returns JWT."""
    import requests
    from config import SUPABASE_URL, SUPABASE_ANON_KEY

    response = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json",
        },
        json={"email": payload.email, "password": payload.password},
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    data = response.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token"),
        "token_type": data["token_type"],
        "expires_in": data["expires_in"],
    }


# =====================================================================
# Protected: PDF Ingestion
# =====================================================================

@app.post("/documents", response_model=IngestionResponse)
def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(rate_limit_dependency),
):
    """
    Upload and ingest a PDF document.

    Protected by JWT auth + rate limiting.
    Parses, chunks, embeds, and stores in Supabase pgvector.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    logger.info(f"User {user.get('sub')} uploading: {file.filename}")

    from ingestion import ingest_pdf
    from rag_graph import clear_bm25_cache

    user_id = user.get("sub")
    result = ingest_pdf(file.file, file.filename, user_id=user_id)

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    # Invalidate cached BM25 index for this user
    clear_bm25_cache(user_id)

    return IngestionResponse(**result)


# =====================================================================
# Protected: RAG Query
# =====================================================================

@app.post("/query", response_model=RAGResponse)
def rag_query(
    request: QueryRequest,
    user: dict = Depends(rate_limit_dependency),
):
    """
    Execute the full RAG pipeline for a user query.

    Pipeline: Input Security → Cache Check → HyDE → Retrieval → RRF →
              Rerank → CRAG → Spotlight → Generate → Self-RAG → Output Security
    """
    query = request.query
    user_id = user.get("sub", "anonymous")
    thread_id = request.thread_id or str(uuid.uuid4())

    # ── Step 1: Input Security Pipeline ──────────────────────────────
    security_result = run_input_security_pipeline(query, max_tokens=MAX_INPUT_TOKENS)

    if not security_result["is_safe"]:
        logger.warning(
            f"Input BLOCKED by {security_result['blocked_by']} for user {user_id}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Your query was blocked by the security pipeline.",
                "blocked_by": security_result["blocked_by"],
                "details": security_result["details"],
            },
        )

    sanitized_query = security_result["sanitized_query"]

    # ── Step 2: Run LangGraph RAG Pipeline ───────────────────────────
    from rag_graph import compile_rag_graph
    global checkpointer

    rag_app = compile_rag_graph(checkpointer=checkpointer)

    initial_state = {
        "user_id": user_id,
        "query": sanitized_query,
        "thread_id": thread_id,
        "cached_answer": None,
        "hyde_answers": [],
        "query_embedding": [],
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "reranked_results": [],
        "web_results": [],
        "crag_decision": "",
        "spotlight_context": "",
        "generated_answer": "",
        "self_rag_score": 0.0,
        "retry_count": 0,
        "final_answer": "",
        "cache_hit": False,
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = rag_app.invoke(initial_state, config=config)

    # ── Step 3: Output Security Pipeline ─────────────────────────────
    raw_answer = final_state.get("final_answer", "No answer generated.")

    output_result = run_output_security_pipeline(raw_answer)
    final_answer = output_result["sanitized_response"]

    # ── Step 4: Pydantic Validation (L9) ─────────────────────────────
    # Build sources list from reranked results
    sources = []
    for doc in final_state.get("reranked_results", []):
        meta = doc.get("metadata", {})
        sources.append({
            "filename": meta.get("filename", "unknown"),
            "page_number": meta.get("page_number", "?"),
            "relevance_score": doc.get("rerank_score", 0),
        })

    # Validate with Pydantic — retry generation once if validation fails
    try:
        response = RAGResponse(
            query=sanitized_query,
            answer=final_answer,
            cache_hit=final_state.get("cache_hit", False),
            sources=sources,
            security_details={
                "input": security_result["details"],
                "output": output_result["details"],
            },
        )
    except Exception as e:
        logger.warning(f"L9 Pydantic validation failed, retrying generation: {e}")
        # Retry: generate a simpler response
        response = RAGResponse(
            query=sanitized_query,
            answer=raw_answer if raw_answer else "An error occurred during response generation.",
            cache_hit=False,
            sources=sources,
            security_details={},
        )

    return response


# =====================================================================
# Protected: User Profile
# =====================================================================

@app.get("/me")
def read_profile(user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return {
        "user_id": user["sub"],
        "email": user.get("email"),
        "role": user.get("role"),
    }


@app.get("/documents/count")
def get_user_document_count(user: dict = Depends(rate_limit_dependency)):
    """Return the number of documents indexed for the logged-in user."""
    import vector_store
    user_id = user.get("sub")
    count = vector_store.get_document_count(user_id=user_id)
    return {"documents_indexed": count}
