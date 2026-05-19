# Suite

The suite page defines the reusable data-layer interfaces that concrete clinical
suites implement.

!!! note "Base classes, not CKD internals"
    This page documents the framework base classes. For the current CKD
    implementation, see Framework Guide -> Suites -> CKD.

## Suite Base Classes

::: krisis.data.base
    options:
      members:
        - BaseDataSuite
        - BasePreprocessor
        - BaseFeatureEngineer
        - BaseGenerator
