"""Shared prompt formatting helpers."""

from __future__ import annotations

from typing import Any


def features_to_bullet_list(features: dict[str, Any]) -> str:
    """Render clinical features as a stable, human-readable bullet list."""
    lines = [f"- {key}: {value}" for key, value in sorted(features.items())]
    return "\n".join(lines)
