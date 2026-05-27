"""Run the CKD detection benchmark.

Usage:
    OPENROUTER_API_KEY=... python examples/ckd_detection.py --model openai/gpt-5.5
    OPENROUTER_API_KEY=... python examples/ckd_detection.py --model anthropic/claude-opus-4.7 --metrics-only
"""

from __future__ import annotations

from _ckd_common import run_ckd_task

from krisis.data.base import Task

if __name__ == "__main__":
    run_ckd_task(Task.DETECTION, "Run the CKD detection benchmark.")
