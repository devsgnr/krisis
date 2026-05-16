"""Run the CKD progression benchmark.

Usage:
    API_KEY=... python examples/ckd_progression.py --backend openai --model gpt-5.2 --n-synthetic 80
    API_KEY=... python examples/ckd_progression.py --backend anthropic --n-synthetic 80 --metrics-only
    API_KEY=... python examples/ckd_progression.py --backend grok --n-synthetic 80 --metrics-only
"""

from __future__ import annotations

from _ckd_common import run_ckd_task

from krisis.data.base import Task

if __name__ == "__main__":
    run_ckd_task(Task.PROGRESSION, "Run the CKD progression benchmark.")
