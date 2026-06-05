# Research Status

Krisis v0.2 is an early evaluation framework for testing LLMs on clinical
reasoning tasks. It is useful for testing safety-relevant model behavior, but it
should be reported with clear scope and limitations.

!!! important "Honest scope"
    Krisis is best understood as an evaluation framework. The CKD suite is the
    first implemented domain, not evidence that the framework covers all of
    clinical medicine.

## What Is Implemented

Krisis v0.2 includes:

| Area | Status |
|---|---|
| CKD suite | Implemented with UCI CKD schema validation |
| Detection task | Implemented |
| Staging task | Implemented using eGFR-derived CKD stage |
| Progression task | Implemented as synthetic stress test |
| API backend | Implemented through OpenRouter-routed model IDs |
| Batching and concurrency | Implemented |
| Retry and fallback behavior | Implemented |
| Abstention-aware metrics | Implemented |
| Calibration metrics | Implemented |
| Text and JSON reports | Implemented |

## What Is Not Claimed

Krisis v0.2 does not claim:

- clinical deployment readiness
- diagnostic approval
- patient-level medical reliability
- real longitudinal CKD progression validation
- broad clinical domain coverage
- replacement for clinician review
- calibrated provider probabilities

## Current Limitations

| Limitation | Why it matters |
|---|---|
| CKD is the only available suite | Cross-domain conclusions are not supported yet |
| UCI CKD is small | Model rankings may be sensitive to small sample behavior |
| UCI CKD is cross-sectional | Real progression cannot be validated from this source |
| Progression is synthetic | It stress-tests reasoning and abstention, not real outcomes |
| Sex is engineered when absent | eGFR needs sex, but UCI CKD does not provide it |
| Confidence is model-reported | It may not be calibrated or comparable across providers |
| Provider APIs evolve | Reproducibility depends on model versions and API behavior |

!!! danger "Do not overclaim"
    Krisis results should not be described as clinical safety certification.
    They are benchmark results under a documented dataset, task, prompt, and
    model configuration.

## Intended Use

Krisis is meant for researchers and engineers who want to evaluate LLM safety
behavior on clinical reasoning tasks:

- whether models answer correctly
- whether they abstain on ambiguous cases
- whether abstention aligns with benchmark risk metadata
- whether stated confidence tracks correctness
- how performance changes across frontier LLM providers

This makes Krisis useful as a human-in-the-loop type system for evaluating
model behavior before AI systems are trusted with high-stakes medical
reasoning.

## Suggested Reporting Practice

When reporting Krisis results, include:

- model provider and exact model ID
- task
- feature set
- dataset source
- seed
- test size
- synthetic record count
- batch size
- max concurrency
- all metrics, not only accuracy
- token total
- elapsed seconds
- whether progression results are synthetic

## Preprint Framing

A careful preprint should frame Krisis as:

- a clinical evaluation framework
- a CKD v0.2 benchmark suite
- an abstention and deferral evaluation tool
- an early research artifact with documented limitations

It should not frame Krisis as:

- a medical device
- a clinical decision support system
- proof that a model is safe for patient care
- evidence of real CKD progression prediction

## Reproducibility Checklist

Before publishing a result table, record:

1. Krisis version or commit hash
2. dataset file/source
3. provider and model ID
4. task
5. feature set
6. seed
7. synthetic count
8. batch size
9. max concurrency
10. full metrics JSON

## Citation

If you use Krisis in research, cite it as software:

```bibtex
@software{watila_krisis_2026,
  author = {Watila, Emmanuel},
  title = {Krisis: A Clinical Evaluation Framework for Large Language Models},
  year = {2026},
  version = {0.2.7},
  url = {https://github.com/devsgnr/krisis}
}
```
