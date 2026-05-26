# Datasets

Datasets define the evidence boundary of a Krisis benchmark. They determine what
clinical variables are available, which labels are trustworthy, which task
claims are valid, and where synthetic stress tests begin.

!!! important "Dataset cards are part of the benchmark"
    LLM evaluation on clinical tasks is only meaningful when the dataset
    boundary is explicit. Krisis documents source, schema, task support, and
    limitations for every suite.

## Available Datasets

| Dataset | Status | Used by | Supported tasks |
|---|---|---|---|
| [CKD Dataset](ckd.md) | Available in v0.2 | `CKDSuite` | detection, staging, synthetic progression |
| [Diabetes](diabetes.md) | Coming soon | Not available yet | Not available yet |
| [Hypertension](hypertension.md) | Coming soon | Not available yet | Not available yet |

## Dataset Policy

Krisis does not bundle clinical datasets in the Python package.

This is intentional:

- clinical datasets can have distribution and licensing constraints
- benchmark users should know exactly which source file was used
- local CSV paths make experiments reproducible without hiding the data source
- package installs stay lightweight

For CKD, users download the UCI dataset separately and pass a local path:

```python
from krisis.data.ckd.suite import CKDSuite

suite = CKDSuite(data_path="datasets/ckd/ckd_full.csv")
```

## What A Dataset Card Should Answer

Each Krisis dataset page should make these points explicit:

| Question | Why it matters |
|---|---|
| Where does the data come from? | Establishes provenance and reproducibility |
| What schema is expected? | Prevents silent misuse of incompatible clinical tables |
| Which tasks are supported? | Avoids claiming more than the source data can support |
| Which fields are engineered? | Separates source observations from benchmark-derived fields |
| Which labels are real vs synthetic? | Keeps detection, staging, and progression claims honest |
| What are the limitations? | Prevents clinical overclaiming |

## Real Data And Synthetic Stress Cases

Krisis can use real rows and synthetic stress cases in the same evaluation set.
For CKD v0.2:

- the real rows come from the UCI CKD dataset
- `n_synthetic` controls completely synthetic patient rows generated from the
  training split
- the staging task is derived from engineered eGFR
- the progression task is synthetic because the source dataset is cross-sectional
- `should_abstain` labels are benchmark metadata, not patient-ground-truth labels

!!! warning "Synthetic does not mean clinical validation"
    Synthetic rows and synthetic progression labels are useful for stress
    testing LLM behavior, especially abstention and deferral. They should not be
    described as real longitudinal clinical outcomes.

!!! note "Synthetic rows are not UCI patients"
    Synthetic benchmark rows are generated from learned per-stage feature
    distributions and clinical bounds. They are useful for controlled stress
    testing, but they are not real patient records and should be reported
    separately from held-out UCI rows.

## Validation Before Evaluation

Dataset validation happens before preprocessing. The CKD suite checks that the
CSV has the expected shape, columns, labels, and value conventions before it
creates benchmark records.

Strong validation helps catch:

- missing required columns
- unexpected columns from a different dataset
- duplicated column names
- non-numeric values in numeric fields
- unsupported categorical values
- missing or duplicated IDs
- target labels that do not include both expected classes

## Adding Future Datasets

New datasets should be added through new suites, not forced into the CKD suite.
That keeps each dataset's units, labels, and clinical assumptions explicit.

A future dataset integration should include:

1. a dataset card
2. a suite-specific validator
3. preprocessing rules
4. feature engineering rules
5. task definitions
6. limitation notes
7. tests for supported schema and task behavior
