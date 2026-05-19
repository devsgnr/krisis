# CKD Dataset

The Krisis CKD suite uses the UCI Machine Learning Repository Chronic Kidney
Disease dataset as its benchmark source.

## Dataset Role

The dataset provides the source clinical rows for:

- CKD detection
- eGFR-derived staging
- synthetic progression stress testing

The dataset is not bundled with the Python package. Users must provide their own
local copy of the UCI CKD CSV via `CKDSuite(data_path="...")`.

## Download

Download the dataset from the UCI Machine Learning Repository:

- [UCI Chronic Kidney Disease dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

Krisis expects a local CSV path:

```python
suite = CKDSuite(data_path="datasets/ckd/ckd_full.csv")
```

## Why The Dataset Is Useful

The UCI CKD dataset contains a mix of:

- renal markers such as serum creatinine and blood urea
- urine markers such as albumin and specific gravity
- hematologic markers such as hemoglobin and packed cell volume
- comorbidity indicators such as hypertension and diabetes
- missing values and categorical fields that require preprocessing

This makes it useful for testing whether LLMs can reason over messy clinical
feature sets rather than clean textbook cases.

## Derived Fields

Krisis derives:

- `sex`: synthetic when the source CSV does not provide it
- `egfr`: CKD-EPI 2021-style estimate
- `ckd_stage`: stage derived from eGFR
- `should_abstain`: deferral-alignment label for ambiguous/conflicting cases

## Feature Engineering Criteria

Krisis engineers CKD features using explicit, documented clinical criteria:

| Engineered field | Criteria used | Reference |
|---|---|---|
| `sex` | Generated when absent using a reproducible serum-creatinine-conditioned heuristic: `sc <= 0.7` biases female, `0.7 < sc <= 0.9` is treated as ambiguous, and `sc > 0.9` biases male. This exists only because the UCI CKD dataset does not include sex. | CKD-EPI creatinine constants and documented Krisis assumption |
| `egfr` | Computed from serum creatinine, age, and sex using the race-free CKD-EPI 2021 creatinine equation. | [National Kidney Foundation CKD-EPI 2021 equation](https://www.kidney.org/ckd-epi-creatinine-equation-2021-0) |
| `ckd_stage` | Derived from eGFR using KDIGO GFR categories: G1 `>=90`, G2 `60-89`, G3 `30-59`, G4 `15-29`, G5 `<15` mL/min/1.73m². Krisis combines G3a/G3b by default, with optional split-stage support in code. | [KDIGO 2024 CKD Guideline](https://kdigo.org/guidelines/ckd-evaluation-and-management/kdigo-2024-ckd-guideline/) |
| `should_abstain` | Derived for evaluation only. Marks cases where eGFR is close to a staging threshold or where the binary CKD label conflicts with the eGFR-derived stage. This label is hidden from the model and used for deferral-alignment scoring. | Krisis benchmark design |

!!! note "Synthetic sex generation"
    The UCI CKD dataset does not include sex. Krisis generates sex using a
    reproducible creatinine-conditioned process so eGFR can be computed. This
    is documented in suite output.

## Included Tasks

- `detection`: CKD vs not CKD
- `staging`: CKD stage classification derived from eGFR
- `progression`: synthetic progression stress test

## Important Limitations

- The dataset is small.
- The dataset is cross-sectional, not longitudinal.
- The progression task is synthetic.
- The dataset is not representative of all CKD populations.
- The benchmark is not clinical validation for patient care.

!!! danger "Do not overclaim"
    Krisis v0.1 evaluates LLM behavior on a CKD benchmark derived from UCI data.
    It does not prove clinical safety, diagnostic performance, or real-world
    deployment readiness.

For suite-level usage and schema requirements, see [CKD Suite](../suites/ckd.md).

## References

- [UCI Chronic Kidney Disease dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- [KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease](https://kdigo.org/guidelines/ckd-evaluation-and-management/kdigo-2024-ckd-guideline/)
- [National Kidney Foundation: CKD-EPI Creatinine Equation (2021)](https://www.kidney.org/ckd-epi-creatinine-equation-2021-0)
