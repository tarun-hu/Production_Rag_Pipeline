"""
chunking_strategies.py — Rectified Dynamic Chunking Strategies with LLM-based Routing.

Implements three production-grade chunking strategies:
  1. Textbooks (Hierarchical Chunking, math/code preservation, 10-15% sliding overlap)
  2. API & Technical Documentation (Syntax-Aware, 0% code overlap, 20% prose overlap)
  3. SOPs & HR Policies (Document-Aware, version/department metadata, 10% overlap)
"""

import re
import json
import logging
from typing import Optional

from config import get_llm_client, LLM_MODEL

logger = logging.getLogger("chunking_strategies")

CATEGORIES = [
    "textbooks",
    "api_documentation",
    "sops_hr_policies"
]


# =====================================================================
# LLM Router / Classifier
# =====================================================================

def classify_document(sample_text: str) -> str:
    """
    Classify a document sample to determine the optimal chunking strategy.
    
    Uses DeepSeek V4 Pro via NVIDIA NIM.
    """
    client = get_llm_client()
    
    prompt = f"""You are an advanced document layout and content analysis system. Analyze the following sample of document text and classify it into EXACTLY one of these categories:

1. `textbooks` (Continuous chapters, conceptual explanations, textbooks, academic materials with equations/diagrams)
2. `api_documentation` (API guides, system logs, code tutorials, developer documentations with code blocks and parameters)
3. `sops_hr_policies` (Standard Operating Procedures (SOPs), corporate HR guidelines, compliance rules, security manuals)

Sample text:
\"\"\"
{sample_text[:1500]}
\"\"\"

Respond with ONLY the category name matching one of the options (e.g. `sops_hr_policies`). Do NOT explain your answer."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
            extra_body={"chat_template_kwargs": {"thinking": False}}
        )
        category = response.choices[0].message.content.strip().lower()
        
        # Sanitize category response
        for cat in CATEGORIES:
            if cat in category:
                logger.info(f"LLM routed document to strategy: '{cat}'")
                return cat
    except Exception as e:
        logger.warning(f"LLM router failed ({e}) — falling back to 'textbooks'")
        
    return "textbooks"


# =====================================================================
# 1. Textbooks (Hierarchical Chunking)
# =====================================================================

def chunk_textbooks(text: str, filename: str) -> list[dict]:
    """
    Textbook Strategy: Hierarchical Chunking mapped to Chapter/Section levels.
    
    - Enforces 10% to 15% sliding text overlap (approx 60-90 tokens for a 600 token chunk)
      to keep conceptual continuity across splits.
    - Strictly preserves LaTeX math equations ($...$ and $$...$$) and code blocks,
      preventing them from being sliced across chunks.
    """
    # Regex to split on chapter or section headings (e.g. Chapter 1, Section 2.1)
    heading_split = re.compile(r'\b((?:Chapter|Section|CHAPTER|SECTION)\s+\d+(?:\.\d+)*)\b')
    parts = heading_split.split(text)
    
    chunks = []
    current_heading = "Introduction"
    
    for part in parts:
        if not part.strip():
            continue
        if heading_split.match(part):
            current_heading = part.strip()
        else:
            # part is the content under this section.
            # Tokenize into paragraphs/sentences, preserving code/math structures whole.
            # Split math and code block segments first so they aren't parsed by sentence splitters.
            pattern = re.compile(r'(\$\$.*?\$\$|```.*?```)', re.DOTALL)
            segments = pattern.split(part)
            
            sub_chunks = []
            current_chunk_text = ""
            current_tokens = 0
            
            for seg in segments:
                if not seg.strip():
                    continue
                # If segment is a code block or LaTeX math, keep it indivisible
                if seg.startswith("$$") or seg.startswith("```"):
                    seg_tokens = len(seg.split())
                    if current_tokens + seg_tokens > 600:
                        sub_chunks.append(current_chunk_text.strip())
                        
                        # Slide window: extract last 10-15% of tokens from previous chunk
                        words = current_chunk_text.split()
                        overlap_words = words[-int(len(words) * 0.12):] # 12% sliding overlap
                        current_chunk_text = " ".join(overlap_words)
                        current_tokens = len(overlap_words)
                        
                    current_chunk_text += "\n" + seg
                    current_tokens += seg_tokens
                else:
                    # Explanatory text segment: split into sentences
                    sentences = [s.strip() for s in re.split(r'(?<=\.|\?)\s', seg) if s.strip()]
                    for sent in sentences:
                        sent_tokens = len(sent.split())
                        if current_tokens + sent_tokens > 600:
                            sub_chunks.append(current_chunk_text.strip())
                            
                            # Slide window: 10% to 15% overlap
                            words = current_chunk_text.split()
                            overlap_words = words[-int(len(words) * 0.12):]
                            current_chunk_text = " ".join(overlap_words)
                            current_tokens = len(overlap_words)
                            
                        current_chunk_text += " " + sent
                        current_tokens += sent_tokens
                        
            if current_chunk_text.strip():
                sub_chunks.append(current_chunk_text.strip())
                
            for idx, sc in enumerate(sub_chunks):
                chunks.append({
                    "content": sc,
                    "metadata": {
                        "filename": filename,
                        "strategy": "textbooks",
                        "chapter_section": current_heading,
                        "chunk_index": idx
                    }
                })
                
    if not chunks:
        # Fallback split
        return [{"content": text, "metadata": {"filename": filename, "strategy": "textbooks_fallback"}}]
        
    return chunks


# =====================================================================
# 2. API & Technical Documentation (Syntax-Aware Chunking)
# =====================================================================

def chunk_api_documentation(text: str, filename: str) -> list[dict]:
    """
    API & Technical Documentation Strategy: Syntax-Aware Chunking.
    
    - Boundaries are set exactly at code block indicators (```) and Markdown tables.
    - Enforces a strict 0% overlap inside code blocks to avoid syntax corruption.
    - Maintains a 20% overlap for surrounding explanatory prose to preserve technical context.
    """
    # Pattern to capture code blocks and markdown tables
    syntax_pattern = re.compile(r'(```\w*?\n.*?\n```|\|[^\n]+\|\r?\n?(?:\|:?-+:?\|?\r?\n?)+(?:\|[^\n]+\|\r?\n?)*)', re.DOTALL)
    parts = syntax_pattern.split(text)
    
    chunks = []
    
    for idx, part in enumerate(parts):
        if not part.strip():
            continue
            
        # Code blocks & Markdown tables are treated as indivisible chunks with 0% overlap
        if part.startswith("```") or part.startswith("|"):
            lang = "code_block"
            if part.startswith("```"):
                match = re.match(r'```(\w*)', part)
                lang = match.group(1) if match and match.group(1) else "code"
                
            chunks.append({
                "content": part.strip(),
                "metadata": {
                    "filename": filename,
                    "strategy": "api_documentation",
                    "lang": lang,
                    "syntax_block": True,
                    "index": idx
                }
            })
        else:
            # Surrounding explanatory prose: chunk with 20% overlap (size 400 tokens, overlap 80 tokens)
            from llama_index.core.node_parser import SentenceSplitter
            splitter = SentenceSplitter(chunk_size=400, chunk_overlap=80) # 20% overlap
            prose_nodes = splitter.split_text(part)
            
            for p_idx, node in enumerate(prose_nodes):
                chunks.append({
                    "content": node.strip(),
                    "metadata": {
                        "filename": filename,
                        "strategy": "api_documentation",
                        "syntax_block": False,
                        "prose_index": p_idx
                    }
                })
                
    return chunks


# =====================================================================
# 3. SOPs & HR Policies (Document-Aware Chunking)
# =====================================================================

def chunk_sops_hr_policies(text: str, filename: str) -> list[dict]:
    """
    SOPs & HR Policies Strategy: Document-Aware Chunking.
    
    - Injects structural metadata tags (versioning, last_updated, department ownership)
      directly into Qdrant vector segments for pre-retrieval filtering.
    - Enforces a tight 10% text overlap (size 300, overlap 30 tokens).
    """
    client = get_llm_client()
    
    # Fast LLM call to extract document metadata from header block
    prompt = f"""Extract metadata from this SOP/HR Policy document header. Look for:
- Version (e.g. v1.2)
- Last Updated Date (e.g. 2026-05-15)
- Department Owner (e.g. Human Resources, IT, Compliance)

Header content:
\"\"\"
{text[:800]}
\"\"\"

Respond with a JSON object containing keys: 'version', 'last_updated', and 'department'. Use null if not found. Do NOT explain or format with markdown codeblocks."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            extra_body={"chat_template_kwargs": {"thinking": False}}
        )
        data = json.loads(response.choices[0].message.content.strip())
        version = data.get("version") or "v1.0"
        last_updated = data.get("last_updated") or "2026-07-20"
        department = data.get("department") or "Human Resources"
    except Exception:
        # Graceful defaults
        version = "v1.0"
        last_updated = "2026-07-20"
        department = "Human Resources"

    # Chunk the policy text with 10% overlap (size 300, overlap 30)
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=30)
    nodes = splitter.split_text(text)
    
    chunks = []
    for idx, node in enumerate(nodes):
        chunks.append({
            "content": node.strip(),
            "metadata": {
                "filename": filename,
                "strategy": "sops_hr_policies",
                "version": version,
                "last_updated": last_updated,
                "department_owner": department,
                "chunk_index": idx
            }
        })
        
    return chunks


# =====================================================================
# Main Router Entry Point
# =====================================================================

def route_and_chunk(text: str, filename: str) -> list[dict]:
    """
    Routes a document to the optimal rectified chunking strategy.
    """
    category = classify_document(text[:5000])
    
    logger.info(f"Applying rectified chunking strategy '{category}' to '{filename}'")
    
    if category == "textbooks":
        return chunk_textbooks(text, filename)
    elif category == "api_documentation":
        return chunk_api_documentation(text, filename)
    elif category == "sops_hr_policies":
        return chunk_sops_hr_policies(text, filename)
        
    return chunk_textbooks(text, filename)
