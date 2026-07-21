"""
security.py — Input & Output Security Pipeline for the Enterprise RAG System.

Layers:
  L1  — Pydantic schema + regex injection detection
  L5  — tiktoken-based input truncation
  L2  — llm-guard prompt injection + toxicity scan
  L7a — PII redaction + content moderation (input)
  L7b — PII redaction + content moderation (output)
"""

import re
import logging
from pydantic import BaseModel, field_validator
from typing import Optional

logger = logging.getLogger("security")

# =====================================================================
# L1 — Pydantic Schema + Regex Injection Detection
# =====================================================================

# Patterns that indicate prompt injection or SQL injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?(your|previous)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if\s+you\s+are",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(system|original)\s+prompt",
    r"\bunion\s+select\b",
    r"\bdrop\s+table\b",
    r"\bdelete\s+from\b",
    r"\binsert\s+into\b",
    r";\s*--",
    r"1\s*=\s*1",
    r"\bexec\s*\(",
    r"\beval\s*\(",
]

COMPILED_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]


class QueryRequest(BaseModel):
    """Pydantic model for incoming RAG query requests."""
    query: str
    thread_id: Optional[str] = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be empty.")
        return v


def check_injection_patterns(text: str) -> tuple[bool, str | None]:
    """
    L1 — Scan text for known injection patterns.

    Returns (is_safe, matched_pattern_or_None).
    """
    for pattern in COMPILED_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            logger.warning(f"L1 INJECTION DETECTED: '{match.group()}' in input")
            return False, match.group()
    return True, None


# =====================================================================
# L5 — Input Restructure / Truncation (tiktoken)
# =====================================================================

def truncate_input(text: str, max_tokens: int = 256, model: str = "gpt-4o") -> str:
    """
    L5 — Truncate input to a maximum number of tokens using tiktoken.

    Protects downstream calls (llm-guard, LLM) from oversized inputs.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            logger.info(f"L5 TRUNCATED: {len(tokens)} tokens → {max_tokens} tokens")
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
    except Exception as e:
        # Fallback: character-based truncation if tiktoken fails
        logger.warning(f"L5 tiktoken fallback (char-based): {e}")
        max_chars = max_tokens * 4  # rough approximation
        if len(text) > max_chars:
            text = text[:max_chars]
    return text


# =====================================================================
# L2 — llm-guard Prompt Injection + Toxicity Scan
# =====================================================================

def scan_with_llm_guard(text: str) -> tuple[str, bool, list[str]]:
    """
    L2 — Run llm-guard scanners on the sanitized input.

    Returns (sanitized_text, is_safe, list_of_triggered_scanner_names).
    Uses lazy imports to avoid slow startup when llm-guard is not needed.
    """
    triggered = []
    try:
        from llm_guard.input_scanners import PromptInjection, Toxicity
        from llm_guard.input_scanners.prompt_injection import MatchType as PIMatchType

        scanners = [
            PromptInjection(threshold=0.5, match_type=PIMatchType.FULL),
            Toxicity(threshold=0.5),
        ]

        sanitized = text
        is_safe = True
        for scanner in scanners:
            sanitized, is_valid, risk_score = scanner.scan("", sanitized)
            if not is_valid:
                is_safe = False
                scanner_name = type(scanner).__name__
                triggered.append(scanner_name)
                logger.warning(
                    f"L2 {scanner_name} TRIGGERED: risk_score={risk_score:.3f}"
                )

        return sanitized, is_safe, triggered

    except ImportError:
        logger.warning("L2 llm-guard not installed — skipping scan")
        return text, True, []
    except Exception as e:
        logger.error(f"L2 llm-guard error: {e}")
        return text, True, []


# =====================================================================
# L7a / L7b — PII Redaction + Content Moderation
# =====================================================================

# PII patterns to redact
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?"      # country code
        r"(?:\(?\d{2,4}\)?[-.\s]?)?"    # area code
        r"\d{3,4}[-.\s]?\d{3,4}"        # local number
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Redact obvious PII from text using regex patterns.

    Returns (redacted_text, list_of_pii_types_found).
    """
    found_types = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found_types.append(pii_type)
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
            logger.info(f"L7 PII REDACTED: {pii_type}")
    return text, found_types



def moderate_content(text: str) -> tuple[bool, dict]:
    """
    Run content through a keyword-based moderation check.

    Returns (is_safe, moderation_result_dict).
    Uses a lightweight approach that doesn't require an external API call.
    """
    # Harmful content keywords to flag
    HARMFUL_CATEGORIES = {
        "violence": [
            r"\b(kill|murder|attack|bomb|weapon|shoot|stab|assault)\b",
        ],
        "self_harm": [
            r"\b(suicide|self[- ]harm|cut myself|end my life)\b",
        ],
        "hate_speech": [
            r"\b(hate|slur|racist|sexist|bigot)\b",
        ],
    }

    flagged_categories = {}
    text_lower = text.lower()

    for category, patterns in HARMFUL_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                flagged_categories[category] = True
                logger.warning(f"L7 MODERATION FLAGGED: {category}")
                break

    if flagged_categories:
        return False, flagged_categories

    return True, {}


# =====================================================================
# Full Input Security Pipeline
# =====================================================================

def run_input_security_pipeline(
    query: str,
    max_tokens: int = 256,
) -> dict:
    """
    Execute the full input security pipeline (L1 → L5 → L2 → L7a).

    Returns a dict with:
      - sanitized_query: the cleaned text
      - is_safe: overall safety verdict
      - blocked_by: which layer blocked it (None if safe)
      - details: additional information from each layer
    """
    details = {}

    # L1 — Regex injection check
    is_safe_l1, matched = check_injection_patterns(query)
    details["l1_injection"] = {"safe": is_safe_l1, "matched": matched}
    if not is_safe_l1:
        return {
            "sanitized_query": query,
            "is_safe": False,
            "blocked_by": "L1_INJECTION_REGEX",
            "details": details,
        }

    # L5 — Truncate
    truncated = truncate_input(query, max_tokens=max_tokens)
    details["l5_truncated"] = len(truncated) < len(query)

    # L7a — PII redaction
    redacted, pii_types = redact_pii(truncated)
    details["l7a_pii_redacted"] = pii_types

    # L2 — llm-guard scan
    sanitized, is_safe_l2, triggered_scanners = scan_with_llm_guard(redacted)
    details["l2_llm_guard"] = {"safe": is_safe_l2, "triggered": triggered_scanners}
    if not is_safe_l2:
        return {
            "sanitized_query": sanitized,
            "is_safe": False,
            "blocked_by": "L2_LLM_GUARD",
            "details": details,
        }

    # L7a — Content moderation
    is_safe_mod, mod_result = moderate_content(sanitized)
    details["l7a_moderation"] = {"safe": is_safe_mod, "flagged": mod_result}
    if not is_safe_mod:
        return {
            "sanitized_query": sanitized,
            "is_safe": False,
            "blocked_by": "L7A_CONTENT_MODERATION",
            "details": details,
        }

    return {
        "sanitized_query": sanitized,
        "is_safe": True,
        "blocked_by": None,
        "details": details,
    }


# =====================================================================
# Output Security Pipeline (L7b + L9)
# =====================================================================

def run_output_security_pipeline(
    response_text: str,
) -> dict:
    """
    Execute the output security pipeline (L7b → L9).

    Returns a dict with:
      - sanitized_response: cleaned response text
      - is_safe: overall safety verdict
      - blocked_by: which layer blocked it (None if safe)
      - details: additional information
    """
    details = {}

    # L7b — PII redaction on output
    redacted, pii_types = redact_pii(response_text)
    details["l7b_pii_redacted"] = pii_types

    # L7b — Content moderation on output
    is_safe_mod, mod_result = moderate_content(redacted)
    details["l7b_moderation"] = {"safe": is_safe_mod, "flagged": mod_result}

    if not is_safe_mod:
        return {
            "sanitized_response": redacted,
            "is_safe": False,
            "blocked_by": "L7B_OUTPUT_MODERATION",
            "details": details,
        }

    return {
        "sanitized_response": redacted,
        "is_safe": True,
        "blocked_by": None,
        "details": details,
    }
