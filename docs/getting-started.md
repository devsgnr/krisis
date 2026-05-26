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
