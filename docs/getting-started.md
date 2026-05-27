# Getting Started

This guide takes you from installation to a first CKD benchmark run.

!!! note "Examples use CKDSuite"
    Most code snippets in the documentation use `CKDSuite` because CKD is the
    only implemented suite in Krisis v0.2. The same framework shape is intended
    for future suites, but diabetes and hypertension are not available yet.

!!! warning "Synthetic benchmark rows"
    When examples set `n_synthetic`, those added patient records are completely
    synthetic benchmark stress cases generated from the training split. They are
    not real UCI patient rows. The progression task is also synthetic because
    the UCI CKD dataset is cross-sectional, not longitudinal.

## Installation

Install Krisis:

```bash
pip install "krisis[api]"
```

Create an API key from [OpenRouter](https://openrouter.ai/settings/keys), then
set it locally:

```bash
export OPENROUTER_API_KEY="..."
```

For experimental local Hugging Face Transformers models, install the `hf` extra
instead:

```bash
pip install "krisis[hf]"
```

## Dataset Setup

Krisis does not bundle the UCI CKD dataset.

Place your local CSV somewhere stable, for example:

```text
datasets/ckd/ckd_full.csv
```

!!! warning "Schema-bound suite"
    `CKDSuite` expects the UCI CKD schema and value conventions. It validates
    required columns, unexpected columns, numeric fields, categorical values,
    unique IDs, and target labels before preprocessing.

## First Benchmark

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

## Running Other Tasks

Change the task in `SuiteConfig`:

```python
SuiteConfig(task=Task.STAGING)
SuiteConfig(task=Task.PROGRESSION)
```

Supported task values:

| Task | Label |
|---|---|
| `Task.DETECTION` | CKD vs not CKD |
| `Task.STAGING` | CKD stage |
| `Task.PROGRESSION` | synthetic stable/worsening/improving trajectory |

## Switching Models

```python
backend = APIBackend(model="anthropic/claude-opus-4.7")
backend = APIBackend(model="x-ai/grok-4.3")
backend = APIBackend(model="google/gemini-3.5-flash")
```

The benchmark code does not change. Only the OpenRouter model id changes.

## Local Transformers

Use `TransformersBackend` when you want to run an open Hugging Face model
locally inside Python.

!!! warning "Experimental backend"
    `TransformersBackend` is experimental in v0.2.2. It defaults to CPU for
    accessibility, but full benchmark runs should use a GPU runtime such as
    Colab or Deepnote with `device="cuda"`.

```python
from krisis.backends.huggingface import TransformersBackend

backend = TransformersBackend(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
    max_new_tokens=512,
)
```

For gated Hugging Face models, set `HF_TOKEN` or pass `hf_token` directly:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

```python
backend = TransformersBackend(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    hf_token="<your-hugging-face-token>",
)
```

GPU runtime:

```python
backend = TransformersBackend(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    device="cuda",
    dtype="bfloat16",
    max_new_tokens=512,
)
```

The same flow is available as an example script:

```bash
python examples/basic_ckd_hf_eval.py --limit 3 --batch-size 1
```

On GPU notebooks, switch devices explicitly:

```bash
python examples/basic_ckd_hf_eval.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --limit 20
```

For gated models in the example runner, either export `HF_TOKEN` first or pass
`--hf-token`.
