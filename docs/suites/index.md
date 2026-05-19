# Suites

Suites are dataset-specific adapters. They turn raw clinical rows into
benchmark-ready `PatientRecord` objects with labels, metadata, prompts, and
suite descriptions.

!!! tip "Suites are the clinical boundary"
    If a dataset uses different units, label definitions, or clinical meaning,
    it should get its own suite rather than being forced into an existing one.

## Available Suites

| Suite | Status | Dataset | Tasks |
|---|---|---|---|
| [CKD Suite](ckd.md) | Available in v0.1 | UCI CKD dataset | detection, staging, synthetic progression |
| [Diabetes Suite](diabetes.md) | Coming soon | Not available yet | Not available yet |
| [Hypertension Suite](hypertension.md) | Coming soon | Not available yet | Not available yet |

## What A Suite Owns

Each suite is responsible for the complete path from source data to model-facing
clinical cases:

| Responsibility | Purpose |
|---|---|
| Dataset loading | Reads the local source file or supported dataset object |
| Schema validation | Rejects unsupported columns, labels, and value formats early |
| Preprocessing | Handles missing values, categorical normalization, scaling, and feature cleanup |
| Feature engineering | Derives clinical fields such as eGFR, CKD stage, or ambiguity labels |
| Task generation | Produces labels for detection, staging, progression, or future tasks |
| Synthetic generation | Adds stress cases where the suite explicitly supports them |
| Metadata reporting | Documents source, feature set, task, record counts, and limitations |

## Suite Configuration

Suites use `SuiteConfig` to keep runs reproducible:

```python
from krisis.data.base import FeatureSet, SuiteConfig, Task
from krisis.data.ckd.suite import CKDSuite

suite = CKDSuite(
    data_path="datasets/ckd/ckd_full.csv",
    config=SuiteConfig(
        task=Task.STAGING,
        features=FeatureSet.FULL,
        n_synthetic=80,
        seed=42,
        test_size=0.2,
    ),
)
```

Key fields:

| Field | Meaning |
|---|---|
| `task` | Which benchmark task to generate |
| `features` | Whether the model sees the full or reduced feature set |
| `n_synthetic` | Number of synthetic stress cases to add |
| `seed` | Random seed for reproducible splitting and generation |
| `test_size` | Fraction of real source data used as held-out evaluation rows |

## The 80/20 Split

For CKD, Krisis uses the configured `test_size` to split the source dataset.
With the default `test_size=0.2`:

- 80% of real rows are used to fit preprocessing and synthetic generation
- 20% of real rows are held out for evaluation
- synthetic rows are generated from the training distribution, not the test rows
- evaluation uses the held-out real rows plus the requested synthetic rows

This keeps the synthetic generator from learning directly from the evaluation
split.

## Suite Output

Calling `suite.load()` returns `PatientRecord` objects. Each record contains:

- model-facing features
- ground-truth label for the selected task
- clinical metadata used for analysis
- `should_abstain` metadata where the task supports deferral scoring

Calling `suite.describe()` returns report metadata, including:

- source dataset
- task
- feature set
- real and synthetic record counts
- label distribution
- stage distribution
- clinical source notes
- suite-specific assumptions

## Adding New Suites

A new suite should be added when a domain has a distinct dataset, schema, task
definition, or clinical interpretation. A good suite should include:

1. a validator for supported source data
2. deterministic preprocessing
3. documented feature engineering
4. task-specific label generation
5. suite metadata for reports
6. focused tests
7. a dataset page and suite page in the docs
