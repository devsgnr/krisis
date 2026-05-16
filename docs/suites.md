# Working With Data Suites

Krisis data suites are dataset-specific adapters. A suite defines how raw rows are
loaded, cleaned, feature-engineered, converted into clinical tasks, and rendered as
`PatientRecord` objects for the benchmark harness.

## CKD Suite

`CKDSuite` is built for the UCI Machine Learning Repository Chronic Kidney
Disease dataset.

For v0.1, `CKDSuite` expects the UCI CKD schema only. Do not pass arbitrary CKD
exports, EHR tables, or custom clinical CSVs directly into this suite unless they
have first been mapped into the UCI CKD column schema and value conventions.

The package does not ship the dataset. Users should download the UCI CKD dataset
themselves and pass a local CSV path:

```python
from krisis.backends.openai import OpenAIBackend
from krisis.benchmark import Benchmark
from krisis.data.base import FeatureSet, SuiteConfig, Task
from krisis.data.ckd.suite import CKDSuite

suite = CKDSuite(
    data_path="/path/to/ckd_full.csv",
    config=SuiteConfig(
        features=FeatureSet.FULL,
        task=Task.PROGRESSION,
        n_synthetic=80,
    ),
)

backend = OpenAIBackend(model="gpt-5.5")
result = Benchmark(suite, backend).run()

print(result.to_json(include_results=False))
```

Example scripts accept the same local path:

```bash
python examples/ckd_progression.py --data-path /path/to/ckd_full.csv --json
```

## Expected CKD Columns

The UCI CKD dataset contains an ID column, clinical feature columns, and a class
label column. Krisis expects the standard UCI feature names and value conventions,
including:

- numeric clinical fields such as `age`, `bp`, `sg`, `al`, `bgr`, `bu`, `sc`,
  `sod`, `pot`, `hemo`, `pcv`, `wbcc`, and `rbcc`
- categorical fields such as `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`,
  `appet`, `pe`, and `ane`
- a target column named `class`, using the UCI CKD labels

Krisis derives eGFR, CKD stage, progression labels, and abstention metadata from
this schema. If another dataset uses different units, label conventions, column
names, or categorical values, the resulting benchmark may be invalid even if the
CSV loads successfully.

## Custom Datasets

Custom clinical datasets should be adapted before use. For now, the recommended
path is:

1. map your dataset into the UCI CKD schema
2. preserve UCI-compatible units and categorical values
3. save the mapped table as a local CSV
4. pass that CSV with `CKDSuite(data_path="...")`

Future versions may add first-class custom dataset adapters, but `CKDSuite` itself
should remain tied to the UCI CKD benchmark definition for reproducibility.
