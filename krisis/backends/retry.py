"""
krisis/backends/retry.py

Small retry helper for transient provider API failures.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_CLASS_FRAGMENTS = (
    "APIConnection",
    "APITimeout",
    "Connection",
    "InternalServer",
    "Overloaded",
    "RateLimit",
    "ServiceUnavailable",
    "Timeout",
)


def is_retryable_exception(exc: BaseException) -> bool:
    """Return True for common transient OpenAI/Anthropic SDK failures."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in RETRYABLE_STATUS_CODES

    name = exc.__class__.__name__
    return any(fragment in name for fragment in RETRYABLE_CLASS_FRAGMENTS)


def call_with_retries(
    operation: Callable[[], T],
    *,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> T:
    """
    Run operation with exponential backoff for transient provider errors.

    ``max_retries=2`` means up to three total attempts.
    """
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_exception(exc):
                raise

            delay = min(max_delay_seconds, base_delay_seconds * (2**attempt))
            if delay > 0:
                jitter = random.uniform(0.0, delay * 0.25)
                time.sleep(delay + jitter)
            attempt += 1
