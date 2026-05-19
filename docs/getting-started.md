# Getting Started

This guide takes you from installation to a first CKD benchmark run.

!!! note "Examples use CKDSuite"
    Most code snippets in the documentation use `CKDSuite` because CKD is the
    only implemented suite in Krisis v0.1. The same framework shape is intended
    for future suites, but diabetes and hypertension are not available yet.

!!! warning "Synthetic benchmark rows"
    When examples set `n_synthetic`, those added patient records are completely
    synthetic benchmark stress cases generated from the training split. They are
    not real UCI patient rows. The progression task is also synthetic because
    the UCI CKD dataset is cross-sectional, not longitudinal.

## Installation

Install Krisis:

```bash
pip install krisis
```

Provider SDKs are optional so users only install what they need:

```bash
pip install "krisis[openai]"
pip install "krisis[anthropic]"
pip install "krisis[grok]"
pip install "krisis[gemini]"
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
from krisis.backends.openai import OpenAIBackend
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

backend = OpenAIBackend(
    model="gpt-5.5",
    api_key="YOUR_API_KEY",
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

## Switching Backends

```python
from krisis.backends.anthropic import AnthropicBackend
from krisis.backends.gemini import GeminiBackend
from krisis.backends.grok import GrokBackend
from krisis.backends.openai import OpenAIBackend
```

All backends work with the same `Benchmark` class and return the same response
fields.
