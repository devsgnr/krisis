"""
krisis/backends/usage.py

Token usage helpers for provider responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Input/output token counts from one provider response."""

    input_tokens: float | None = None
    output_tokens: float | None = None

    @property
    def total_tokens(self) -> float | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0.0) + (self.output_tokens or 0.0)


def _usage_value(usage: Any, *names: str) -> float | None:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return float(value)
    return None


def usage_from_openai_compatible_response(response: Any) -> TokenUsage:
    """Extract token usage from OpenAI-compatible response objects."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=_usage_value(usage, "prompt_tokens", "input_tokens"),
        output_tokens=_usage_value(usage, "completion_tokens", "output_tokens"),
    )


def usage_from_anthropic_response(response: Any) -> TokenUsage:
    """Extract token usage from Anthropic Messages API response objects."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=_usage_value(usage, "input_tokens"),
        output_tokens=_usage_value(usage, "output_tokens"),
    )


def usage_from_gemini_response(response: Any) -> TokenUsage:
    """Extract token usage from Google Gemini response objects."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=_usage_value(
            usage,
            "prompt_token_count",
            "input_token_count",
            "input_tokens",
        ),
        output_tokens=_usage_value(
            usage,
            "candidates_token_count",
            "output_token_count",
            "output_tokens",
        ),
    )
