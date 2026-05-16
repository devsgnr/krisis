"""Run the CKD detection benchmark.

Usage:
    API_KEY=... python examples/ckd_detection.py --backend openai --model gpt-5.2
    API_KEY=... python examples/ckd_detection.py --backend anthropic --metrics-only
    API_KEY=... python examples/ckd_detection.py --backend grok --metrics-only
"""

from __future__ import annotations

from _ckd_common import run_ckd_task

from krisis.data.base import Task

if __name__ == "__main__":
    run_ckd_task(Task.DETECTION, "Run the CKD detection benchmark.")
