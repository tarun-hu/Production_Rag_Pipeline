"""
rate_limiter.py — Upstash sliding-window rate limiter for FastAPI.

Chains after auth so we rate-limit by authenticated user ID.
"""

from fastapi import Depends, HTTPException, status
from upstash_redis import Redis
from upstash_ratelimit import Ratelimit, SlidingWindow

from config import (
    UPSTASH_REDIS_REST_URL,
    UPSTASH_REDIS_REST_TOKEN,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from auth import get_current_user

# ── Upstash Redis client ─────────────────────────────────────────────
redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)

# ── Rate limiter: sliding window ─────────────────────────────────────
ratelimit = Ratelimit(
    redis=redis,
    limiter=SlidingWindow(
        max_requests=RATE_LIMIT_MAX_REQUESTS,
        window=RATE_LIMIT_WINDOW_SECONDS,
    ),
    prefix="ratelimit",
)


def rate_limit_dependency(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    FastAPI dependency: checks rate limit for the authenticated user.

    Must be chained *after* get_current_user so we have a user ID.
    Returns the user payload if allowed; raises HTTP 429 if over limit.
    """
    user_id = user.get("sub", "anonymous")
    try:
        response = ratelimit.limit(user_id)
        if not response.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {response.reset - int(__import__('time').time())} seconds.",
                headers={"Retry-After": str(response.reset - int(__import__('time').time()))},
            )
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("rate_limiter").warning(
            f"Upstash rate limiter unavailable ({e}). Allowing request to proceed."
        )
        
    return user
