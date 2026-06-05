# Krisis

[![PyPI](https://img.shields.io/pypi/v/krisis.svg)](https://pypi.org/project/krisis/)
[![PyPI downloads](https://static.pepy.tech/badge/krisis)](https://pepy.tech/project/krisis)

Clinical evaluation framework for testing LLM safety behavior in medical reasoning.

Krisis evaluates not only whether an LLM is correct, but whether it knows when
to abstain, defer, or express uncertainty in high-stakes clinical tasks.

## Why Krisis

Krisis grew out of [Cady AI](https://x.com/devsgnr_/status/1996591411441852793?s=20), an earlier CKD
detection chatbot presented at a national AI hackathon. Cady AI used a model
trained on the UCI Chronic Kidney Disease dataset to predict CKD/not-CKD,
return class probabilities, and attribute which lab results pushed risk upward.

That project exposed the next safety question: as LLMs become more fluent in
clinical reasoning, can they recognize cases where they should not confidently
answer? Krisis turns that question into a reusable evaluation framework: a
human-in-the-loop type system for checking whether LLMs can defer,
abstain, and express uncertainty before their outputs are trusted.

## What Krisis Does

Krisis provides:

- clinical task suites that produce structured patient records
- a unified API backend for OpenAI, Anthropic, Grok, Gemini, and other OpenRouter-routed models
- batched and concurrent benchmark execution
- retry/backoff handling for transient provider failures
- structured parsing of model predictions, confidence, and abstentions
- abstention-aware metrics beyond plain accuracy
- text, full JSON, and metrics-only JSON reports
- execution metadata such as runtime, throughput, batch size, concurrency, and token usage

## Research Status And Limitations

Krisis v0.2 currently includes one implemented suite: Chronic Kidney Disease
(CKD), based on the UCI CKD dataset.

Supported CKD tasks:

- `detection`: CKD vs not CKD
- `staging`: CKD stage classification
- `progression`: synthetic progression stress test

Important limitations:

- CKD is the only available suite in v0.2.
- The UCI CKD dataset is small and cross-sectional.
- Progression is synthetic because the source dataset is not longitudinal.
- Krisis is for research and evaluation only. It is not a medical device and
  must not be used to diagnose or treat patients.
- Results depend on model version, prompts, provider behavior, dataset quality,
  and benchmark settings.

## Installation

Install Krisis:

```bash
pip install krisis
```

Install API model support:

```bash
pip install "krisis[api]"
```

Then create an API key from
[OpenRouter](https://openrouter.ai/settings/keys) and set it locally:

```bash
export OPENROUTER_API_KEY="..."
```

Install experimental local Hugging Face Transformers support:

```bash
pip install "krisis[hf]"
```

## Quickstart

> **Warning**
> Krisis v0.2 only includes the CKD suite. The UCI CKD CSV is not bundled with
> the package; download it locally and pass its path to `CKDSuite`.

```python
from krisis.backends.api import APIBackend
from krisis.benchmark import Benchmark
from krisis.data.base import FeatureSet, SuiteConfig, Task
from krisis.data.ckd.suite import CKDSuite
from krisis.results.report import format_report

suite = CKDSuite(
    config=SuiteConfig(
        features=FeatureSet.FULL,
        task=Task.DETECTION,
        seed=42,
        n_synthetic=80,
        test_size=0.2,
    ),
    data_path="datasets/ckd/ckd_full.csv",
)

backend = APIBackend(
    model="openai/gpt-5.5",
    api_key="YOUR_OPENROUTER_API_KEY",
    reasoning_effort="low",
)

result = Benchmark(
    suite,
    backend,
    batch_size=8,
    max_concurrency=2,
).run()

print(format_report(result))
```

## Outputs

Krisis supports three report styles.

Text report:

```python
from krisis.results.report import format_report

print(format_report(result))
```

Full JSON report:

```python
from krisis.results.report import format_json_report

print(format_json_report(result, include_results=True))
```

Metrics-only JSON report for plotting/model comparison:

```python
from krisis.results.report import format_metrics_json_report

print(format_metrics_json_report(result))
```

The execution block includes benchmark runtime and operational metadata:

```json
{
  "batch_size": 8,
  "max_concurrency": 2,
  "n_input_records": 160,
  "n_api_batches": 20,
  "elapsed_seconds": 42.18,
  "records_per_second": 3.79,
  "input_tokens": 12000,
  "output_tokens": 2400,
  "token_total": 14400
}
```

## Core Concepts

- **Suite**: prepares a clinical dataset/task and returns patient records.
- **Backend**: adapts a model provider to Krisis' standard response shape.
- **Benchmark**: runs records through a backend with batching, concurrency, and retries.
- **Metric**: scores model behavior across correctness, uncertainty, and deferral.
- **Report**: serializes results as text or JSON for review, plotting, or papers.

## Metrics

Krisis includes:

- Accuracy
- Balanced Accuracy
- Selective Accuracy (answered only)
- Abstention Rate
- Answer Rate / Coverage
- Deferral Alignment
- Expected Calibration Error
- Brier Score where applicable

Selective accuracy separates how often the model was right when it answered
from how often it chose not to answer.

## Model Backends

| Route | Backend | Example model |
|---|---|---|
| API | `APIBackend` | `openai/gpt-5.5` |
| API | `APIBackend` | `anthropic/claude-opus-4.7` |
| API | `APIBackend` | `x-ai/grok-4.3` |
| API | `APIBackend` | `google/gemini-3.5-flash` |
| experimental local HF | `TransformersBackend` | `Qwen/Qwen2.5-0.5B-Instruct` |

`TransformersBackend` is experimental in v0.2.7. It is meant for GPU notebooks
and local experimentation; CPU runs are useful for smoke tests but too slow for
serious benchmark runs.

The Hugging Face backend supports **causal text-generation models only**:
models loadable with `AutoModelForCausalLM`. Classifier, embedding,
masked-language, seq2seq, and multimodal-only models are outside this backend's
scope and will raise an initialization error.

For gated Hugging Face models, set `HF_TOKEN` or pass `hf_token` directly:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

```python
backend = TransformersBackend(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    hf_token="<your-hugging-face-token>",
)
```

All backends return the same structured fields:

```python
prediction
abstained
confidence
raw_response
input_tokens
output_tokens
total_tokens
```

Run a CKD smoke test with a local Transformers model:

```bash
python examples/basic_ckd_hf_eval.py --limit 3 --batch-size 1
```

Use a GPU runtime such as Colab or Deepnote by passing `--device cuda`.

## Citation

If you use Krisis in research, please cite it as software:

```bibtex
@software{watila_krisis_2026,
  author = {Watila, Emmanuel},
  title = {Krisis: A Clinical Evaluation Framework for Large Language Models},
  year = {2026},
  version = {0.2.7},
  url = {https://github.com/devsgnr/krisis}
}
```

## License

Apache-2.0
