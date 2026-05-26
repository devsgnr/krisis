"""Run the CKD progression benchmark.

Usage:
    OPENROUTER_API_KEY=... python examples/ckd_progression.py --model openai/gpt-5.5 --n-synthetic 80
    OPENROUTER_API_KEY=... python examples/ckd_progression.py --model anthropic/claude-4.7-opus --n-synthetic 80 --metrics-only
"""

from __future__ import annotations

from _ckd_common import run_ckd_task

from krisis.data.base import Task

if __name__ == "__main__":
    run_ckd_task(Task.PROGRESSION, "Run the CKD progression benchmark.")
