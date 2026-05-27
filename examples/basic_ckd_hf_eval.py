"""Run any CKD benchmark task with an experimental Transformers backend.

Install:
    pip install "krisis[hf]"

CPU smoke test:
    python examples/basic_ckd_hf_eval.py --limit 3 --batch-size 1

GPU example:
    python examples/basic_ckd_hf_eval.py \
        --model-id Qwen/Qwen2.5-7B-Instruct \
        --device cuda \
        --dtype bfloat16 \
        --limit 20

DeepSeek-style models that require remote code:
    python examples/basic_ckd_hf_eval.py \
        --model-id deepseek-ai/DeepSeek-R1 \
        --device cuda \
        --trust-remote-code \
        --limit 5

Gated models:
    export HF_TOKEN=<your-hugging-face-token>
    python examples/basic_ckd_hf_eval.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --device cuda \
        --limit 5
"""

from __future__ import annotations

from _ckd_common import build_ckd_hf_parser, run_ckd_hf_benchmark

from krisis.data.base import FeatureSet, Task


def main() -> None:
    parser = build_ckd_hf_parser("Run a CKD benchmark with Transformers.")
    args = parser.parse_args()

    run_ckd_hf_benchmark(
        task=Task(args.task),
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.hf_token,
        limit=args.limit,
        n_synthetic=args.n_synthetic,
        features=FeatureSet(args.features),
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        as_json=args.json,
        metrics_only=args.metrics_only,
        include_results=args.include_results,
    )


if __name__ == "__main__":
    main()
