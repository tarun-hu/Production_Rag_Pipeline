"""
ingestion.py — PDF Ingestion Pipeline with dynamic layout-aware chunking.

Parses PDFs, combines pages with layout markers, classifies content via LLM,
routes to the optimal chunking strategy, embeds locally, and stores in Qdrant.
"""

import logging
import os
import re
import tempfile
from typing import BinaryIO

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

logger = logging.getLogger("ingestion")


def parse_pdf(file: BinaryIO, filename: str) -> list[dict]:
    """
    Parse a PDF file and return a list of page-level text documents.

    Uses LlamaIndex's PDFReader for robust text extraction.
    Returns: [{"text": "...", "page_number": 1, "filename": "doc.pdf"}, ...]
    """
    from llama_index.core import SimpleDirectoryReader
    from llama_index.readers.file import PDFReader

    # Save uploaded file to a temp directory for LlamaIndex to read
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "wb") as f:
            content = file.read()
            f.write(content)

        # Use SimpleDirectoryReader which handles PDF extraction
        reader = SimpleDirectoryReader(
            input_dir=tmpdir,
            file_extractor={".pdf": PDFReader()},
        )
        llama_docs = reader.load_data()

    documents = []
    for idx, doc in enumerate(llama_docs, start=1):
        page_num = str(doc.metadata.get("page_label") or doc.metadata.get("page_number") or idx)
        documents.append({
            "text": doc.text,
            "page_number": page_num,
            "filename": filename,
        })

    logger.info(f"Parsed {len(documents)} pages from '{filename}'")
    return documents


def post_process_chunks(chunks: list[dict], combined_text: str) -> list[dict]:
    """
    Post-process chunks to:
      1. Extract exact page numbers based on the position of page markers in combined_text.
      2. Strip [PAGE_X] layout markers from the final chunk text.
    """
    page_positions = []
    for match in re.finditer(r'\[PAGE_(\d+)\]', combined_text):
        page_positions.append((match.group(1), match.start()))
        
    for chunk in chunks:
        content = chunk["content"]
        # Clean page markers out of chunk content first
        cleaned_content = re.sub(r'\[PAGE_\d+\]', '', content).strip()
        
        # Take a robust snippet of the cleaned content (first 50 chars) to locate position in combined_text
        snippet = cleaned_content[:50].strip()
        start_pos = -1
        
        if snippet:
            start_pos = combined_text.find(snippet)
            if start_pos == -1 and len(cleaned_content) > 100:
                # Try middle snippet fallback if head snippet had formatting alterations
                mid = len(cleaned_content) // 2
                snippet_mid = cleaned_content[mid:mid+50].strip()
                if snippet_mid:
                    start_pos = combined_text.find(snippet_mid)
        
        page_num = "1"
        if start_pos != -1 and page_positions:
            # Find the last page marker that starts before or at start_pos
            for num, pos in page_positions:
                if pos <= start_pos:
                    page_num = num
                else:
                    break
        
        chunk["content"] = cleaned_content
        
        # Set final metadata
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["page_number"] = page_num

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed each chunk's content locally using BAAI/bge-small-en-v1.5.

    Adds an 'embedding' key to each chunk dict.
    """
    import models_local

    model = models_local.get_embedding_model()

    # Collect all texts
    texts = [chunk["content"] for chunk in chunks]

    # Generate embeddings locally in a single batch (runs on GPU if available)
    logger.info(f"Computing local embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    # Attach embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info(f"Embedded {len(chunks)} chunks locally using {model.device}")
    return chunks


def ingest_pdf(file: BinaryIO, filename: str, user_id: str) -> dict:
    """
    Full ingestion pipeline: parse → compile → LLM route → chunk → embed → store.

    Returns a summary dict with counts and status.
    """
    import vector_store
    import chunking_strategies

    # Step 1: Parse page-by-page
    documents = parse_pdf(file, filename)
    if not documents:
        return {
            "status": "error",
            "message": "No text could be extracted from the PDF.",
            "filename": filename,
            "pages": 0,
            "chunks": 0,
        }

    # Step 2: Combine pages with page markers to preserve boundaries
    combined_text = ""
    for doc in documents:
        combined_text += f"\n[PAGE_{doc['page_number']}]\n{doc['text']}"

    # Step 3: LLM classify and route to optimal chunking strategy
    logger.info(f"Routing document '{filename}' to optimal chunker...")
    raw_chunks = chunking_strategies.route_and_chunk(combined_text, filename)
    
    # Post-process to assign page numbers and remove layout markers
    chunks = post_process_chunks(raw_chunks, combined_text)

    # Step 4: Embed (locally on GPU/CPU)
    chunks = embed_chunks(chunks)

    # Step 5: Store in local Qdrant collection isolated by user_id
    inserted = vector_store.add_documents(chunks, user_id)

    result = {
        "status": "success",
        "filename": filename,
        "pages": len(documents),
        "chunks": inserted,
        "message": f"Successfully ingested '{filename}': {len(documents)} pages → {inserted} chunks indexed.",
    }
    logger.info(result["message"])
    return result
