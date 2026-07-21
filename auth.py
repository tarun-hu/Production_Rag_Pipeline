"""
auth.py — Supabase JWT Authentication dependency for FastAPI.

Fetches and caches Supabase's public signing keys (JWKS),
then verifies every incoming Bearer token against them.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient

from config import SUPABASE_URL, JWKS_URL

# ── Bearer scheme for Swagger / OpenAPI docs ─────────────────────────
bearer_scheme = HTTPBearer()

SUPABASE_AUDIENCE = "authenticated"

# ── Lazy JWKS client (initialized on first use) ─────────────────────
_jwks_client = None


def _get_jwks_client() -> PyJWKClient:
    """Lazily initialize the JWKS client so missing config doesn't crash at import."""
    global _jwks_client
    if _jwks_client is None:
        if not SUPABASE_URL:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SUPABASE_URL is not configured. Set it in .env.",
            )
        _jwks_client = PyJWKClient(JWKS_URL)
    return _jwks_client


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency: extracts and verifies a Supabase-issued JWT.

    Returns the decoded payload dict on success.
    Raises HTTP 401 on any verification failure.
    """
    token = credentials.credentials
    try:
        jwks_client = _get_jwks_client()
        # Look up the correct public key using the token's "kid" header
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience=SUPABASE_AUDIENCE,
        )
    except HTTPException:
        raise  # Re-raise our own 503
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
        )
    return payload
