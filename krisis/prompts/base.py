"""
krisis/prompts/base.py

Dispatch for task-specific chat prompts.
"""

from __future__ import annotations

from krisis.data.base import PatientRecord, Task
from krisis.prompts.detection import build_detection_messages
from krisis.prompts.progression import build_progression_messages
from krisis.prompts.staging import build_staging_messages


def build_messages(record: PatientRecord, task: Task) -> list[dict[str, str]]:
    """
    Build OpenAI-style chat messages for the given patient row and task.

    Each inner dict has keys: role ("system" | "user"), content (str).
    """
    if task == Task.DETECTION:
        return build_detection_messages(record)
    if task == Task.STAGING:
        return build_staging_messages(record)
    if task == Task.PROGRESSION:
        return build_progression_messages(record)
    raise ValueError(f"Unsupported task: {task!r}")
