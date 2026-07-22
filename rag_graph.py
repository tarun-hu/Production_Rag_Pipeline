"""
rag_graph.py — LangGraph State Machine for the RAG Pipeline.

Nodes:
  1. check_answer_cache  — skip pipeline if cached answer exists
  2. hyde                — generate hypothetical answers
  3. embed_query         — embed query + HyDE outputs (with cache)
  4. hybrid_retrieval    — dense (pgvector) + sparse (BM25) search
  5. rrf_fusion          — reciprocal rank fusion (k=60)
  6. rerank              — cross-encoder reranking (flashrank)
  7. crag_check          — grade relevance, fallback to Tavily
  8. spotlight           — wrap chunks in XML delimiters
  9. generate            — LLM answer generation (NVIDIA NIM / DeepSeek)
  10. self_rag_reflect   — grade answer quality, retry if needed
  11. cache_answer       — store answer in cache
"""

import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from config import (
    CRAG_RELEVANCE_THRESHOLD,
    SELF_RAG_QUALITY_THRESHOLD,
    SELF_RAG_MAX_RETRIES,
    RERANK_TOP_K,
    HYDE_NUM_HYPOTHETICALS,
    LLM_MODEL,
    TAVILY_API_KEY,
)

logger = logging.getLogger("rag_graph")

# Phrases that indicate the LLM deflected / refused to answer from the context
_DEFLECTION_PATTERNS = [
    "does not contain",
    "do not contain",
    "doesn't contain",
    "don't contain",
    "no information about",
    "not mentioned in",
    "no relevant information",
    "cannot answer",
    "unable to answer",
    "not enough information",
    "no specific information",
    "not addressed in",
    "not covered in",
    "not available in the provided",
    "beyond the scope of the provided",
]

def _is_deflection(answer: str) -> bool:
    """Return True if the generated answer is a deflection / refusal."""
    lowered = answer.lower()
    return any(p in lowered for p in _DEFLECTION_PATTERNS)


# =====================================================================
# State Definition
# =====================================================================

class RAGState(TypedDict):
    """State that flows through the RAG pipeline graph."""
    # Input
    user_id: str
    query: str
    thread_id: str

    # Cache
    cached_answer: Optional[str]

    # HyDE
    hyde_answers: list[str]

    # Embeddings
    query_embedding: list[float]

    # Retrieval
    dense_results: list[dict]
    sparse_results: list[dict]
    fused_results: list[dict]
    reranked_results: list[dict]

    # CRAG / HITL
    web_results: list[dict]
    crag_decision: str  # "accept" or "web_fallback"
    web_search_needed: bool
    web_search_approved: Optional[bool]

    # Context
    spotlight_context: str

    # Generation
    generated_answer: str
    self_rag_score: float
    retry_count: int

    # Final
    final_answer: str
    cache_hit: bool


# =====================================================================
# Node Implementations
# =====================================================================

def check_answer_cache(state: RAGState) -> dict:
    """Check if a cached answer exists for this query."""
    from cache import get_cached_answer

    cached = get_cached_answer(state["query"], user_id=state["user_id"])
    if cached:
        logger.info("RAG ANSWER CACHE HIT — skipping full pipeline")
        return {
            "cached_answer": cached,
            "final_answer": cached,
            "cache_hit": True,
        }
    return {"cached_answer": None, "cache_hit": False}


def hyde_node(state: RAGState) -> dict:
    """Generate hypothetical answers using HyDE technique via NVIDIA NIM."""
    from config import get_llm_client

    client = get_llm_client()
    query = state["query"]

    prompt = f"""You are an expert assistant. Generate {HYDE_NUM_HYPOTHETICALS} hypothetical answers 
to the following question. Each answer should be a realistic, detailed paragraph as if it were 
extracted from a real document. Separate each answer with '---'.

Question: {query}

Hypothetical Answers:"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
        raw = response.choices[0].message.content
        answers = [a.strip() for a in raw.split("---") if a.strip()]
    except Exception as e:
        logger.warning(f"HyDE LLM generation failed ({e}) — falling back to empty list.")
        answers = []

    logger.info(f"HyDE generated {len(answers)} hypothetical answers")
    return {"hyde_answers": answers}


def embed_query_node(state: RAGState) -> dict:
    """Embed the query using the cached embedding wrapper."""
    from cache import get_embedding

    query_emb = get_embedding(state["query"])
    logger.info("Query embedded (with cache check)")
    return {"query_embedding": query_emb}


# ── BM25 Caching (In-Memory per user) ─────────────────────────────────
_bm25_cache = {}
_corpus_cache = {}

def clear_bm25_cache(user_id: str):
    """Invalidate a user's cached BM25 index when they upload a new PDF."""
    global _bm25_cache, _corpus_cache
    if user_id in _bm25_cache:
        _bm25_cache.pop(user_id, None)
    if user_id in _corpus_cache:
        _corpus_cache.pop(user_id, None)
    logger.info(f"Invalidated in-memory BM25 cache for user: {user_id}")


def hybrid_retrieval_node(state: RAGState) -> dict:
    """
    Perform hybrid retrieval:
      - Dense: cosine similarity via local Qdrant (filtered by user_id, k=25)
      - Sparse: BM25 index cached in memory per user_id (k=25)
    """
    import vector_store
    from rank_bm25 import BM25Okapi

    query = state["query"]
    query_emb = state["query_embedding"]
    user_id = state["user_id"]

    # Dense retrieval via local Qdrant with user filtering (top 25)
    dense = vector_store.search(query_emb, user_id=user_id, k=25, threshold=0.0)

    # Sparse retrieval via BM25 over user's documents
    global _bm25_cache, _corpus_cache
    
    if user_id in _bm25_cache:
        bm25 = _bm25_cache[user_id]
        all_docs = _corpus_cache[user_id]
        logger.info(f"BM25 INDEX CACHE HIT for user: {user_id}")
    else:
        # Cache miss: pull documents and build index in memory
        logger.info(f"BM25 INDEX CACHE MISS for user: {user_id}. Fetching documents...")
        all_docs = vector_store.get_all_documents(user_id=user_id)
        if all_docs:
            corpus = [doc["content"].lower().split() for doc in all_docs]
            bm25 = BM25Okapi(corpus)
            _bm25_cache[user_id] = bm25
            _corpus_cache[user_id] = all_docs
        else:
            bm25 = None
            _corpus_cache[user_id] = []

    sparse = []
    if bm25 and all_docs:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top 25 by BM25 score
        scored_docs = list(zip(all_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        sparse = [
            {**doc, "bm25_score": float(score)}
            for doc, score in scored_docs[:25]
            if score > 0
        ]

    logger.info(f"Hybrid retrieval: {len(dense)} dense + {len(sparse)} sparse results")
    return {"dense_results": dense, "sparse_results": sparse}


def rrf_fusion_node(state: RAGState) -> dict:
    """
    Reciprocal Rank Fusion (RRF) with k=60 constant.
    Combines dense and sparse search results into a unified candidate set.
    """
    dense = state.get("dense_results", [])
    sparse = state.get("sparse_results", [])

    k_constant = 60
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank + 1))

    for rank, doc in enumerate(sparse):
        doc_id = doc["id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank + 1))

    fused = []
    for doc_id, score in rrf_scores.items():
        d = doc_map[doc_id].copy()
        d["rrf_score"] = score
        fused.append(d)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    logger.info(f"RRF fusion produced {len(fused)} merged results")
    return {"fused_results": fused}


def rerank_node(state: RAGState) -> dict:
    """
    Cross-Encoder Reranking using BAAI/bge-reranker-base.
    Calculates precise relevance scores on GPU via PyTorch FP16 autocast.
    """
    import models_local
    import torch

    candidates = state.get("fused_results", [])
    query = state["query"]

    if not candidates:
        logger.info("Rerank: No candidates to rerank.")
        return {"reranked_results": []}

    ranker = models_local.get_reranker_model()

    # Form query-document pairs
    pairs = [[query, doc["content"]] for doc in candidates]
    
    # Compute relevance scores locally on GPU (autocast FP16) or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            scores = ranker.predict(pairs, batch_size=32)
    else:
        scores = ranker.predict(pairs, batch_size=32)

    # Map scores back and sort candidates descending
    results = []
    for doc, score in zip(candidates, scores):
        d = doc.copy()
        d["rerank_score"] = float(score)
        results.append(d)

    # Sort descending by CrossEncoder score and take top K
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    results = results[:RERANK_TOP_K]

    logger.info(f"Reranked to top {len(results)} results using local CrossEncoder on {device}")
    return {"reranked_results": results}


def crag_check_node(state: RAGState) -> dict:
    """
    CRAG: Grade relevance of top result.
    Determines if a web search is needed.
    """
    results = state["reranked_results"]

    if not results:
        logger.info("CRAG: No local documents found — web search needed.")
        return {
            "crag_decision": "web_fallback",
            "web_search_needed": True,
            "web_results": [],
        }

    top_score = results[0].get("rerank_score", 0)
    if top_score < CRAG_RELEVANCE_THRESHOLD:
        logger.info(f"CRAG: Top score {top_score:.3f} < threshold {CRAG_RELEVANCE_THRESHOLD} — web search needed.")
        return {
            "crag_decision": "web_fallback",
            "web_search_needed": True,
            "web_results": [],
        }

    logger.info(f"CRAG: Top score {top_score:.3f} — accepting local results.")
    return {
        "crag_decision": "accept",
        "web_search_needed": False,
        "web_results": [],
    }


def web_fallback_node(state: RAGState) -> dict:
    """
    Node to execute Tavily web search.
    Pauses before execution via LangGraph interrupt.
    """
    logger.info(f"web_fallback_node called. State keys: {list(state.keys())}")
    approved = state.get("web_search_approved")
    logger.info(f"web_fallback_node: approved is {approved} (type: {type(approved)})")
    
    if approved is True:
        logger.info("Web search APPROVED. Executing Tavily query...")
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(query=state["query"], max_results=3)
            web_results = [
                {
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                }
                for r in response.get("results", [])
            ]
            logger.info(f"Tavily returned {len(web_results)} web results")
            return {"web_results": web_results}
        except Exception as e:
            logger.error(f"Tavily fallback failed: {e}")
            return {"web_results": []}
    else:
        # If there's already a generated answer (deflection-triggered HITL)
        # show that answer instead of the generic upload warning.
        existing_answer = state.get("generated_answer")
        if existing_answer:
            logger.info("Web search REJECTED. Returning the existing generated answer.")
            msg = existing_answer
        else:
            logger.info("Web search REJECTED. No documents — showing upload warning.")
            msg = "You haven't embedded any documents yet, upload your files to get started."
        return {
            "final_answer": msg,
            "generated_answer": msg,
            "web_results": [],
        }


def spotlight_node(state: RAGState) -> dict:
    """Wrap retrieved chunks in XML delimiters for LLM grounding."""
    parts = []

    # Use reranked results or web results depending on CRAG decision
    if state.get("crag_decision") == "web_fallback" and state.get("web_results"):
        for i, result in enumerate(state["web_results"]):
            parts.append(
                f'<web_result index="{i+1}" url="{result.get("url", "")}">\n'
                f'{result["content"]}\n'
                f'</web_result>'
            )
    else:
        for i, result in enumerate(state.get("reranked_results", [])):
            meta = result.get("metadata", {})
            source = meta.get("filename", "unknown")
            page = meta.get("page_number", "?")
            parts.append(
                f'<document index="{i+1}" source="{source}" page="{page}">\n'
                f'{result["content"]}\n'
                f'</document>'
            )

    context = "\n\n".join(parts) if parts else "<no_context>No relevant documents found.</no_context>"
    logger.info(f"Spotlight: wrapped {len(parts)} context items in XML")
    return {"spotlight_context": context}


def generate_node(state: RAGState) -> dict:
    """Generate the final answer grounded on retrieved context via NVIDIA NIM."""
    from config import get_llm_client

    client = get_llm_client()

    system_prompt = (
        "You are an expert, highly intelligent RAG assistant. Your job is to provide a rich, detailed, "
        "and well-structured answer to the user's question by carefully analyzing ALL provided context chunks below.\n\n"
        "Instructions:\n"
        "1. Synthesize concepts, definitions, formulas, real-world applications, and explanations across ALL context chunks provided.\n"
        "2. Do NOT give up or say 'no information' if the topic is present in any chunk. Extract every relevant detail, example, and derivation.\n"
        "3. Focus deeply on the main terms of the user's query.\n"
        "4. Organize your response clearly with headings, bullet points, and citations (source filename and page number).\n"
        "5. If a chunk contains questions or exercises (e.g. 'Find applications...'), use your scientific reasoning grounded in the surrounding context chunks to answer them thoroughly."
    )

    user_prompt = (
        f"Context:\n{state['spotlight_context']}\n\n"
        f"Question: {state['query']}\n\n"
        f"Answer:"
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            top_p=0.95,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Generation LLM call failed: {e}")
        answer = "The primary LLM service (NVIDIA DeepSeek NIM) is currently experiencing a temporary outage or credit limits. Please check your credentials or try again later."

    logger.info(f"Generated answer ({len(answer)} chars)")
    return {"generated_answer": answer}


def self_rag_reflect_node(state: RAGState) -> dict:
    """
    Self-RAG: Grade the generated answer's groundedness.
    If below threshold, increment retry count for regeneration.
    """
    from config import get_llm_client

    client = get_llm_client()
    retry_count = state.get("retry_count", 0)

    grading_prompt = f"""You are a strict grading assistant. Evaluate the following answer 
for groundedness and quality based on the provided context.

Context:
{state['spotlight_context']}

Question: {state['query']}

Answer: {state['generated_answer']}

Score the answer from 0.0 to 1.0 where:
- 1.0 = perfectly grounded, accurate, and complete
- 0.0 = completely ungrounded, hallucinated, or wrong

Respond with ONLY a single decimal number (e.g., 0.85). Nothing else."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": grading_prompt}],
            temperature=0.0,
            top_p=0.95,
            max_tokens=16,
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
        score = float(response.choices[0].message.content.strip())
    except Exception as e:
        logger.warning(f"Self-RAG grading LLM call failed ({e}) — defaulting score to 1.0 (pass)")
        score = 1.0  # Default to pass if grading is down to avoid infinite loops

    logger.info(f"Self-RAG score: {score:.3f} (threshold={SELF_RAG_QUALITY_THRESHOLD}, retry={retry_count})")

    # ── Deflection Detection ────────────────────────────────────────
    # If the LLM said "the context doesn't contain...", the retrieved
    # documents were topically irrelevant.  Route to web search HITL
    # instead of accepting or retrying this non-answer.
    # BUT: skip this check if web search was already approved & executed —
    # web search is the last resort, so accept whatever it produces.
    if _is_deflection(state["generated_answer"]) and not state.get("web_search_approved"):
        logger.info("Self-RAG: Deflection detected in generated answer — routing to web search HITL.")
        return {
            "self_rag_score": score,
            "retry_count": retry_count,
            "web_search_needed": True,
            "crag_decision": "web_fallback",
        }

    if score >= SELF_RAG_QUALITY_THRESHOLD or retry_count >= SELF_RAG_MAX_RETRIES:
        return {
            "self_rag_score": score,
            "final_answer": state["generated_answer"],
            "retry_count": retry_count,
            "web_search_needed": False,
        }
    else:
        return {
            "self_rag_score": score,
            "retry_count": retry_count + 1,
        }


def cache_answer_node(state: RAGState) -> dict:
    """Store the final answer in the RAG answer cache."""
    from cache import set_cached_answer

    # Do not cache fallback LLM error response, upload warnings, or deflection answers
    err_msg = "The primary LLM service (NVIDIA DeepSeek NIM) is currently experiencing"
    warn_msg = "You haven't embedded any documents yet"
    
    if err_msg in state["final_answer"] or warn_msg in state["final_answer"]:
        logger.info("Skipping caching for fallback LLM outage or upload warning response.")
        return {}

    if _is_deflection(state["final_answer"]):
        logger.info("Skipping caching for deflection/refusal answer.")
        return {}

    set_cached_answer(state["query"], state["final_answer"], user_id=state["user_id"])
    return {}


# =====================================================================
# Routing Functions
# =====================================================================

def route_after_cache_check(state: RAGState) -> str:
    """If cache hit, skip straight to output. Otherwise, run pipeline."""
    if state.get("cache_hit"):
        return "cache_answer"  # Skip to end
    return "hyde"


def route_after_self_rag(state: RAGState) -> str:
    """If a final answer is set, proceed to cache.
    If web_search_needed is set (deflection or CRAG), route to web_fallback.
    Otherwise, regenerate."""
    if state.get("final_answer"):
        return "cache_answer"
    if state.get("web_search_needed"):
        return "web_fallback"
    return "generate"


def route_after_crag(state: RAGState) -> str:
    """Route to web fallback if search is needed, else proceed to spotlight."""
    if state.get("web_search_needed"):
        return "web_fallback"
    return "spotlight"


def route_after_web_fallback(state: RAGState) -> str:
    """If web search was rejected, skip generation and go to caching. Else proceed to spotlight."""
    val = state.get("web_search_approved")
    logger.info(f"route_after_web_fallback: web_search_approved is {val} (type: {type(val)})")
    if val is False:
        return "cache_answer"
    return "spotlight"


# =====================================================================
# Graph Construction
# =====================================================================

def build_rag_graph():
    """Build and compile the full RAG pipeline LangGraph."""

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("check_cache", check_answer_cache)
    graph.add_node("hyde", hyde_node)
    graph.add_node("embed_query", embed_query_node)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
    graph.add_node("rrf_fusion", rrf_fusion_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("crag_check", crag_check_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("spotlight", spotlight_node)
    graph.add_node("generate", generate_node)
    graph.add_node("self_rag_reflect", self_rag_reflect_node)
    graph.add_node("cache_answer", cache_answer_node)

    # Edges
    graph.add_edge(START, "check_cache")
    graph.add_conditional_edges("check_cache", route_after_cache_check)
    graph.add_edge("hyde", "embed_query")
    graph.add_edge("embed_query", "hybrid_retrieval")
    graph.add_edge("hybrid_retrieval", "rrf_fusion")
    graph.add_edge("rrf_fusion", "rerank")
    graph.add_edge("rerank", "crag_check")
    graph.add_conditional_edges("crag_check", route_after_crag)
    graph.add_conditional_edges("web_fallback", route_after_web_fallback)
    graph.add_edge("spotlight", "generate")
    graph.add_edge("generate", "self_rag_reflect")
    graph.add_conditional_edges("self_rag_reflect", route_after_self_rag)
    graph.add_edge("cache_answer", END)

    return graph


def compile_rag_graph(checkpointer=None):
    """Build, compile, and return the runnable RAG graph with HITL interrupts before web search."""
    graph = build_rag_graph()
    
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
    return graph.compile(checkpointer=checkpointer, interrupt_before=["web_fallback"])
