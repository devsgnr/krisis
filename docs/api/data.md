# Data

The data page contains the shared Pydantic value objects used by suites,
backends, metrics, and benchmark runs.

`PatientRecord` and `SuiteConfig` validate inputs at construction time. This is
especially useful for public suite APIs because invalid synthetic counts,
invalid test splits, and malformed record shapes fail early.

## Enums And Records

::: krisis.data.base
    options:
      members:
        - FeatureSet
        - Task
        - PatientRecord
        - SuiteConfig
