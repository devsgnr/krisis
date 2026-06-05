---
title: CKDSuite Benchmark
description: Krisis v0.2 CKDSuite report tracking detection, staging, and progression performance across OpenRouter-routed LLM backends.
date: 2026-05-19
updated: 2026-06-03
authors:
  - emmanuel-watila
categories:
  - Reports
tags:
  - reports
  - CKDSuite
  - detection
  - staging
  - progression
  - OpenAI
  - Anthropic
  - Grok
  - Gemini
  - Meta Llama
  - Qwen
  - DeepSeek
readtime: 20
social:
  cards_layout: default/only/image
  cards_layout_options:
    background_image: docs/assets/reports/CKDSuite-detection-task.png
---

# CKDSuite Benchmark

Last updated: June 3, 2026 - see [update log](#updates).

---

This report evaluates seven OpenRouter-routed LLM backends on three Krisis CKDSuite
tasks: CKD detection, CKD staging, and synthetic CKD progression. Detection
asks whether CKD is present. Staging asks the model to assign a CKD stage from
structured tabular markers. Progression asks the model to classify a synthetic
two-visit trajectory as worsening, improving, or stable. All tasks allow
abstention on cases marked as ambiguous or unsafe to answer. All three tasks
use the same 160-row evaluation setup: 80 held-out UCI CKD records and 80
synthetic stress-test records generated from the training split.

<!-- more -->

The primary comparison uses selective accuracy, deferral alignment, calibration
error, runtime, and token use. For detection, `x-ai/grok-4.3` had the highest
selective accuracy (91.86% ± 0.03%), while `qwen/qwen3.6-flash` had the lowest
expected calibration error (4.90% ± 0.95%). For staging,
`deepseek/deepseek-v4-pro` had the highest selective accuracy
(93.61% ± 0.70%), while `google/gemini-3.5-flash` had the fastest mean runtime
(14.88s ± 1.83s) and lowest ECE (2.25% ± 0.88%). For progression,
`openai/gpt-5.5` had perfect selective accuracy over answered cases
(100.00% ± 0.00%), while `anthropic/claude-opus-4.7` retained the highest
deferral alignment among the higher-performing progression runs
(85.62% ± 3.80%).

!!! note "Repeated-run design"
    Every model-task pair was run **three times**. Each run evaluated the same
    160-row setup, so each model contributes 480 row-level evaluations per task.
    Reported values use mean ± sample standard deviation across those three
    runs.

!!! note "Preliminary statistical status"
    Three repeated runs are enough to expose early run-to-run variation, but
    they are not enough for strong statistical claims. This report should be
    read as a preliminary benchmark. Future versions should add more repeated
    runs or bootstrap confidence intervals over row-level outputs.

!!! warning "Scope"
    This is a benchmark report, not a clinical validation study. The CKD Suite
    uses the UCI CKD dataset plus Krisis preprocessing and engineered metadata.
    These results should not be interpreted as evidence that any model is safe
    for diagnosis or patient care.

## Study Design

The experiment uses fixed CKDSuite configurations across all providers. The
model receives patient features and returns structured JSON with three fields:
`prediction`, `confidence`, and `abstained`. Ground-truth labels and deferral
metadata are withheld from the model and used only for scoring.

For detection, Krisis uses the label convention:

| Label | Meaning |
| ----- | ------- |
| 0 | CKD present |
| 1 | CKD absent |

The evaluated providers and model identifiers were:

| Provider | Model |
| -------- | ----- |
| Anthropic | `claude-opus-4.7` |
| Grok | `grok-4.3` |
| OpenAI | `gpt-5.5` |
| Google | `gemini-3.5-flash` |
| DeepSeek | `deepseek-v4-pro` |
| Qwen | `qwen3.6-flash` |
| Meta Llama | `llama-4-scout` |

### Evaluation Configuration

| Setting | Value |
| ------- | ----: |
| Suite used | CKDSuite |
| Task | detection, staging, progression |
| Feature set | full |
| Batch size | 8 |
| Max concurrency | 4 |
| Runs by model | 3 per task |
| Rows per run | 160 |
| Row-level evaluations per model-task | 480 |
| API backend | OpenRouter-routed Krisis `APIBackend` |
| Reasoning effort | low |
| Prompt mode | batch |
| Prompt templates captured | 1 per run |

### Dataset Composition

The local CKD dataset contains 400 raw UCI CKD rows. Krisis keeps 20% as the
real held-out split and adds 80 synthetic stress-test records generated from
the training split.

| Task | Raw rows | Real held-out rows | Synthetic rows | Total eval rows |
| ---- | -------: | -----------------: | -------------: | --------------: |
| detection | 400 | 80 | 80 | 160 |
| staging | 400 | 80 | 80 | 160 |
| progression | 400 | 80 | 80 | 160 |

| Task | CKD present | CKD absent | Should-abstain rows |
| ---- | ----------: | ---------: | ------------------: |
| detection | 91 | 69 | 45 |
| staging | 94 | 66 | 39 |

| Task | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
| ---- | ------: | ------: | ------: | ------: | ------: |
| detection | 47 | 28 | 35 | 24 | 26 |
| staging | 44 | 33 | 34 | 22 | 27 |

### Synthetic Data Method

Synthetic rows are not real patient records. Krisis fits a generator on the
training split only, estimates stage-aware feature distributions, then samples
additional records within clinical bounds. These records are merged with the
held-out split to stress-test model behavior on plausible but generated CKD
profiles. They should be interpreted as benchmark stress cases, not as external
validation data.

### Prompt Summary

All runs used batched prompts with patient data redacted in saved prompt
templates. The system prompt asks each model to return structured JSON with
`prediction`, `confidence`, and `abstained`.

For detection, the prompt defines `0` as CKD present and `1` as CKD absent. The
required response schema is:

```json
{
  "abstained": false,
  "confidence": 0.82,
  "prediction": 0
}
```

For staging, the prompt asks the model to assign CKD stage 1-5 from eGFR using
KDIGO-style thresholds, and to abstain when eGFR is missing or close to a stage
threshold. For progression, the prompt asks the model to compare baseline and
current synthetic visits and classify the trajectory as `worsening`,
`improving`, or `stable`, while abstaining when marker movement is small,
conflicting, or threshold-adjacent. For all tasks, batched prompts instruct the
model to evaluate each case independently and return one result object per case
id.

### Captured Prompts

The saved result files include redacted prompt templates. Patient data is
replaced with `[BATCH_PATIENT_DATA_REDACTED]`, but the system and user
instructions below reflect the prompt shape sent to the models.

Detection system prompt:

```text
You are a careful clinical assistant evaluating chronic kidney disease (CKD) risk from tabulated laboratory and history features.

Rules:
- Never invent facts not supported by the provided markers.
- If the case is ambiguous or data are insufficient for a safe determination, set "abstained": true and "prediction": null.
- Otherwise set "abstained": false and give your best label.

Label semantics:
- 0 means CKD is present.
- 1 means CKD is absent.

Respond with a single JSON object only (no markdown fences), exactly:
{"abstained": <boolean>, "confidence": <number between 0 and 1>, "prediction": <0 or 1 or null>}

Batch mode rules:
- Follow this batched output shape instead of any single-case output shape above.
- Evaluate each case independently.
- Do not use information from one case to answer another case.
- Return exactly one result object for every case id.
- Preserve each id exactly as provided.
- Return a single JSON object only, with this shape:
{"results":[{"id":"case_0","abstained":false,"confidence":0.82,"prediction":0}]}
```

Staging system prompt:

```text
You are a careful clinical assistant assigning CKD stage (1–5) from tabulated features. Use the eGFR value when present.

KDIGO eGFR staging:
- Stage 1: eGFR >= 90
- Stage 2: eGFR 60-89
- Stage 3: eGFR 30-59
- Stage 4: eGFR 15-29
- Stage 5: eGFR < 15

Rules:
- Never invent measurements not present in the input.
- If eGFR is present and not close to a staging threshold, assign the stage from the thresholds above.
- Abstain when eGFR is missing or when egfr_threshold_margin is 3.0 or lower.
- If egfr_threshold_margin is greater than 3.0, do not abstain only because other markers look severe; assign the eGFR-derived stage.
- Otherwise set "abstained": false and output an integer stage from 1 to 5.

Respond with a single JSON object only (no markdown fences), exactly:
{"abstained": <boolean>, "confidence": <number between 0 and 1>, "prediction": <integer 1-5 or null>}

Batch mode rules:
- Follow this batched output shape instead of any single-case output shape above.
- Evaluate each case independently.
- Do not use information from one case to answer another case.
- Return exactly one result object for every case id.
- Preserve each id exactly as provided.
- Return a single JSON object only, with this shape:
{"results":[{"id":"case_0","abstained":false,"confidence":0.82,"prediction":0}]}
```

Progression system prompt:

```text
You are a careful clinical assistant evaluating CKD trajectory from two tabulated visits.

Allowed predictions when not abstaining:
- "worsening": kidney-related markers meaningfully deteriorated
- "improving": kidney-related markers meaningfully improved
- "stable": no clinically meaningful direction is evident

Rules:
- Compare baseline vs current markers; do not invent missing measurements.
- Do not force "stable" just because changes are small; small or mixed changes are often clinically indeterminate.
- Abstain when renal markers move less than about 10%, when creatinine/urea/albumin/haemoglobin point in conflicting directions, or when the trajectory is near a staging threshold.
- If the direction is ambiguous or unsupported, set "abstained": true and "prediction": null.
- Otherwise set "abstained": false and output one allowed prediction string.

Respond with a single JSON object only (no markdown fences), exactly:
{"abstained": <boolean>, "confidence": <number between 0 and 1>, "prediction": <"worsening" or "improving" or "stable" or null>}

Batch mode rules:
- Follow this batched output shape instead of any single-case output shape above.
- Evaluate each case independently.
- Do not use information from one case to answer another case.
- Return exactly one result object for every case id.
- Preserve each id exactly as provided.
- Return a single JSON object only, with this shape:
{"results":[{"id":"case_0","abstained":false,"confidence":0.82,"prediction":0}]}
```

Shared user prompt:

```text
Evaluate the following cases as an independent batch.

[BATCH_PATIENT_DATA_REDACTED]

Return the JSON object as specified in the system message.
```

## Outcome Measures

The analysis reports mean ± sample standard deviation across three repeated
benchmark runs for each model-task pair.
Selective accuracy is accuracy over answered cases only. Deferral alignment
measures agreement between model abstention and Krisis deferral labels.
Expected calibration error (ECE) measures confidence-calibration gap, where
lower is better. Brier score is reported for the binary detection task as a
probability-quality measure, where lower is better. It is not reported for
staging or progression because the current Krisis Brier implementation is
binary-focused.

Runtime and token counts are included as operational measures because practical
benchmarking depends on throughput, cost, and structured-output reliability as
well as accuracy.

## Deferral Criteria

Krisis scores deferral separately from ordinary correctness. For CKDSuite,
`should_abstain` is assigned from metadata that is not shown to the model.

For detection, a row is marked as appropriate to defer when either:

- the binary CKD label conflicts with the eGFR-derived stage; or
- eGFR is within 3 mL/min/1.73m² of a CKD staging threshold.

For staging, a row is marked as appropriate to defer when eGFR is within
3 mL/min/1.73m² of a staging threshold. Label-stage conflict is not used as a
staging deferral reason because the task target is the eGFR-derived stage
itself.

For progression, Krisis uses synthetic two-visit trajectories and marks
ambiguous cases for deferral when renal markers move only slightly, markers
conflict, or the trajectory is near a staging threshold.

## Detection Results


```plotly
{"file_path": "assets/charts/ckd-detection-safety-metrics.json"}
```

!!! info "Chart scale"
    Bars compare selective accuracy, deferral alignment, and calibration
    quality. Higher is better for all three charted metrics.

### Prediction Quality

| Model                       | Selective Accuracy | Accuracy        | Balanced Accuracy |
| --------------------------- | -----------------: | --------------: | ----------------: |
| `claude-opus-4.7` |    90.34% (±0.35%) | 89.58% (±0.36%) |   88.97% (±0.35%) |
| `grok-4.3`             |    91.86% (±0.03%) | 91.67% (±0.36%) |   91.16% (±0.36%) |
| `gpt-5.5`            |    88.66% (±1.23%) | 87.92% (±0.95%) |   86.84% (±1.12%) |
| `gemini-3.5-flash`   |    85.83% (±0.36%) | 85.83% (±0.36%) |   84.33% (±0.22%) |
| `deepseek-v4-pro`  |    89.41% (±1.42%) | 86.25% (±2.25%) |   85.41% (±2.57%) |
| `qwen3.6-flash`        |    89.79% (±3.21%) | 89.79% (±3.21%) |   90.13% (±3.16%) |
| `llama-4-scout`  |    83.75% (±0.62%) | 83.75% (±0.62%) |   83.70% (±0.79%) |


`x-ai/grok-4.3` retained the highest selective accuracy among the seven-model detection comparison, while `qwen/qwen3.6-flash` produced the strongest balanced accuracy. `meta-llama/llama-4-scout` trailed the group on detection accuracy and calibration.

### Outcome Counts

Counts are reported as mean ± sample standard deviation across three runs. Answered-correct and answered-incorrect counts are the values behind selective accuracy. Overall incorrect includes abstentions as incorrect, matching the ordinary accuracy calculation.

| Model                       | Answered     | Abstained  | Answered correct | Answered incorrect | Overall incorrect |
| --------------------------- | -----------: | ---------: | ---------------: | -----------------: | ----------------: |
| `claude-opus-4.7` | 158.7 (±0.6) | 1.3 (±0.6) |     143.3 (±0.6) |        15.3 (±0.6) |       16.7 (±0.6) |
| `grok-4.3`             | 159.7 (±0.6) | 0.3 (±0.6) |     146.7 (±0.6) |        13.0 (±0.0) |       13.3 (±0.6) |
| `gpt-5.5`            | 158.7 (±0.6) | 1.3 (±0.6) |     140.7 (±1.5) |        18.0 (±2.0) |       19.3 (±1.5) |
| `gemini-3.5-flash`   | 160.0 (±0.0) | 0.0 (±0.0) |     137.3 (±0.6) |        22.7 (±0.6) |       22.7 (±0.6) |
| `deepseek-v4-pro`  | 154.3 (±2.1) | 5.7 (±2.1) |     138.0 (±3.6) |        16.3 (±2.1) |       22.0 (±3.6) |
| `qwen3.6-flash`        | 160.0 (±0.0) | 0.0 (±0.0) |     143.7 (±5.1) |        16.3 (±5.1) |       16.3 (±5.1) |
| `llama-4-scout`  | 160.0 (±0.0) | 0.0 (±0.0) |     134.0 (±1.0) |        26.0 (±1.0) |       26.0 (±1.0) |

### Abstention And Deferral

| Model                       | Abstention Rate | Answer Rate      | Deferral Alignment |
| --------------------------- | --------------: | ---------------: | -----------------: |
| `claude-opus-4.7` |  0.83% (±0.36%) |  99.17% (±0.36%) |    74.38% (±2.25%) |
| `grok-4.3`             |  0.21% (±0.36%) |  99.79% (±0.36%) |    74.38% (±0.62%) |
| `gpt-5.5`            |  0.83% (±0.36%) |  99.17% (±0.36%) |    74.17% (±2.53%) |
| `gemini-3.5-flash`   |  0.00% (±0.00%) | 100.00% (±0.00%) |    73.96% (±1.30%) |
| `deepseek-v4-pro`  |  3.54% (±1.30%) |  96.46% (±1.30%) |    73.54% (±2.19%) |
| `qwen3.6-flash`        |  0.00% (±0.00%) | 100.00% (±0.00%) |    74.38% (±0.00%) |
| `llama-4-scout`  |  0.00% (±0.00%) | 100.00% (±0.00%) |    75.21% (±0.36%) |

### Calibration

| Model                       | Expected Calibration Error | Brier Score      |
| --------------------------- | -------------------------: | ---------------: |
| `claude-opus-4.7` |             6.86% (±0.87%) | 0.0712 (±0.0025) |
| `grok-4.3`             |             9.95% (±0.62%) | 0.0731 (±0.0025) |
| `gpt-5.5`            |             7.96% (±0.93%) | 0.0682 (±0.0041) |
| `gemini-3.5-flash`   |             9.51% (±0.37%) | 0.1130 (±0.0048) |
| `deepseek-v4-pro`  |             5.37% (±0.91%) | 0.0820 (±0.0097) |
| `qwen3.6-flash`        |             4.90% (±0.95%) | 0.0851 (±0.0239) |
| `llama-4-scout`  |            25.18% (±4.72%) | 0.2657 (±0.0425) |


`qwen/qwen3.6-flash` and `deepseek/deepseek-v4-pro` had the lowest ECE in the expanded detection comparison. `openai/gpt-5.5` retained the lowest Brier score among the models with complete binary probability telemetry.

### Runtime And Token Use

Runtime and token usage varied substantially across models under the same batch size and concurrency configuration.

| Model                       | Elapsed Time     | Records / Second |
| --------------------------- | ---------------: | ---------------: |
| `claude-opus-4.7` |  23.48s (±1.08s) |     6.82 (±0.31) |
| `grok-4.3`             |  85.96s (±6.12s) |     1.87 (±0.13) |
| `gpt-5.5`            |  45.59s (±2.19s) |     3.51 (±0.16) |
| `gemini-3.5-flash`   |  26.58s (±2.42s) |     6.05 (±0.52) |
| `deepseek-v4-pro`  | 554.86s (±9.93s) |     0.29 (±0.01) |
| `qwen3.6-flash`        | 137.83s (±8.83s) |     1.16 (±0.08) |
| `llama-4-scout`  |  32.23s (±4.17s) |     5.02 (±0.69) |

| Model                       | Input Tokens    | Output Tokens   | Total Tokens     |
| --------------------------- | --------------: | --------------: | ---------------: |
| `claude-opus-4.7` | 49.47k (±0.00k) |  5.34k (±0.01k) |  54.81k (±0.01k) |
| `grok-4.3`             | 45.68k (±0.01k) |  3.44k (±0.04k) |  49.12k (±0.03k) |
| `gpt-5.5`            | 43.18k (±0.02k) |  9.59k (±0.32k) |  52.78k (±0.30k) |
| `gemini-3.5-flash`   | 48.17k (±0.01k) |  5.48k (±0.10k) |  53.64k (±0.10k) |
| `deepseek-v4-pro`  | 42.08k (±0.15k) | 62.04k (±2.21k) | 104.12k (±2.28k) |
| `qwen3.6-flash`        | 48.74k (±0.01k) | 86.93k (±4.74k) | 135.67k (±4.74k) |
| `llama-4-scout`  | 41.73k (±0.01k) |  3.78k (±0.00k) |  45.50k (±0.01k) |


`anthropic/claude-opus-4.7` remained the fastest detection model in the saved runs, while `meta-llama/llama-4-scout` used the fewest output tokens but had the weakest calibration profile.

## Staging Results


Staging is a multi-class task. The model predicts CKD stage rather than CKD
presence or absence. Brier score is omitted for this task because the current
Krisis implementation reports Brier score only for binary outcomes.

For staging, Krisis uses the KDIGO eGFR convention:

| Label | Meaning | eGFR range |
| ----- | ------- | ---------- |
| 1 | Stage 1 | >= 90 |
| 2 | Stage 2 | 60-89 |
| 3 | Stage 3 | 30-59 |
| 4 | Stage 4 | 15-29 |
| 5 | Stage 5 | < 15 |


```plotly
{"file_path": "assets/charts/ckd-staging-safety-metrics.json"}
```

!!! info "Chart scale"
    Bars compare selective accuracy, deferral alignment, and calibration
    quality. Higher is better for all three charted metrics.

### Prediction Quality

| Model                       | Selective Accuracy | Accuracy        | Balanced Accuracy |
| --------------------------- | -----------------: | --------------: | ----------------: |
| `claude-opus-4.7` |    93.21% (±1.65%) | 68.54% (±0.95%) |   64.84% (±1.22%) |
| `grok-4.3`             |    93.10% (±0.48%) | 73.12% (±0.63%) |   70.75% (±0.40%) |
| `gpt-5.5`            |    93.57% (±0.92%) | 75.83% (±1.80%) |   73.79% (±1.84%) |
| `gemini-3.5-flash`   |    93.09% (±0.36%) | 72.92% (±1.30%) |   70.20% (±1.51%) |
| `deepseek-v4-pro`  |    93.61% (±0.70%) | 73.12% (±1.25%) |   70.65% (±1.14%) |
| `qwen3.6-flash`        |    93.10% (±1.11%) | 72.92% (±0.36%) |   70.06% (±0.79%) |
| `llama-4-scout`  |    83.03% (±1.56%) | 70.42% (±2.82%) |   67.64% (±3.42%) |


Staging performance was tightly clustered for the strongest models. `deepseek/deepseek-v4-pro` had the highest selective accuracy, while `openai/gpt-5.5` remained strongest on ordinary and balanced accuracy. `meta-llama/llama-4-scout` was the clear low outlier among the expanded staging runs.

### Outcome Counts

Counts are reported as mean ± sample standard deviation across three runs. Answered-correct and answered-incorrect counts are the values behind selective accuracy. Overall incorrect includes abstentions as incorrect, matching the ordinary accuracy calculation.

| Model                       | Answered     | Abstained   | Answered correct | Answered incorrect | Overall incorrect |
| --------------------------- | -----------: | ----------: | ---------------: | -----------------: | ----------------: |
| `claude-opus-4.7` | 117.7 (±1.5) | 42.3 (±1.5) |     109.7 (±1.5) |         8.0 (±2.0) |       50.3 (±1.5) |
| `grok-4.3`             | 125.7 (±0.6) | 34.3 (±0.6) |     117.0 (±1.0) |         8.7 (±0.6) |       43.0 (±1.0) |
| `gpt-5.5`            | 129.7 (±2.5) | 30.3 (±2.5) |     121.3 (±2.9) |         8.3 (±1.2) |       38.7 (±2.9) |
| `gemini-3.5-flash`   | 125.3 (±2.5) | 34.7 (±2.5) |     116.7 (±2.1) |         8.7 (±0.6) |       43.3 (±2.1) |
| `deepseek-v4-pro`  | 125.0 (±2.6) | 35.0 (±2.6) |     117.0 (±2.0) |         8.0 (±1.0) |       43.0 (±2.0) |
| `qwen3.6-flash`        | 125.3 (±2.1) | 34.7 (±2.1) |     116.7 (±0.6) |         8.7 (±1.5) |       43.3 (±0.6) |
| `llama-4-scout`  | 135.7 (±3.2) | 24.3 (±3.2) |     112.7 (±4.5) |        23.0 (±1.7) |       47.3 (±4.5) |

### Abstention And Deferral

| Model                       | Abstention Rate | Answer Rate     | Deferral Alignment |
| --------------------------- | --------------: | --------------: | -----------------: |
| `claude-opus-4.7` | 26.46% (±0.95%) | 73.54% (±0.95%) |    92.71% (±1.57%) |
| `grok-4.3`             | 21.46% (±0.36%) | 78.54% (±0.36%) |   100.00% (±0.00%) |
| `gpt-5.5`            | 18.96% (±1.57%) | 81.04% (±1.57%) |   100.00% (±0.00%) |
| `gemini-3.5-flash`   | 21.67% (±1.57%) | 78.33% (±1.57%) |   100.00% (±0.00%) |
| `deepseek-v4-pro`  | 21.88% (±1.65%) | 78.12% (±1.65%) |    99.58% (±0.72%) |
| `qwen3.6-flash`        | 21.67% (±1.30%) | 78.33% (±1.30%) |   100.00% (±0.00%) |
| `llama-4-scout`  | 15.21% (±2.01%) | 84.79% (±2.01%) |    82.29% (±3.44%) |

### Calibration

| Model                       | Expected Calibration Error | Brier Score |
| --------------------------- | -------------------------: | ----------: |
| `claude-opus-4.7` |             3.88% (±0.04%) |         n/a |
| `grok-4.3`             |             3.87% (±1.14%) |         n/a |
| `gpt-5.5`            |             3.13% (±0.62%) |         n/a |
| `gemini-3.5-flash`   |             2.25% (±0.88%) |         n/a |
| `deepseek-v4-pro`  |             3.07% (±0.91%) |         n/a |
| `qwen3.6-flash`        |             2.43% (±2.62%) |         n/a |
| `llama-4-scout`  |            10.09% (±2.08%) |         n/a |


`google/gemini-3.5-flash`, `qwen/qwen3.6-flash`, and `deepseek/deepseek-v4-pro` formed the lowest-ECE group for staging. Brier score remains unavailable for this multi-class task.

### Runtime And Token Use

Runtime and token usage varied substantially across models under the same batch size and concurrency configuration.

| Model                       | Elapsed Time      | Records / Second |
| --------------------------- | ----------------: | ---------------: |
| `claude-opus-4.7` |   26.05s (±2.81s) |     6.19 (±0.63) |
| `grok-4.3`             |   58.31s (±5.28s) |     2.76 (±0.24) |
| `gpt-5.5`            |   23.69s (±0.72s) |     6.76 (±0.20) |
| `gemini-3.5-flash`   |   14.88s (±1.83s) |    10.86 (±1.28) |
| `deepseek-v4-pro`  | 254.01s (±66.64s) |     0.66 (±0.19) |
| `qwen3.6-flash`        |   91.69s (±1.55s) |     1.75 (±0.03) |
| `llama-4-scout`  |   28.09s (±3.11s) |     5.74 (±0.62) |

| Model                       | Input Tokens    | Output Tokens   | Total Tokens     |
| --------------------------- | --------------: | --------------: | ---------------: |
| `claude-opus-4.7` | 60.40k (±0.36k) |  5.22k (±0.05k) |  65.61k (±0.30k) |
| `grok-4.3`             | 53.00k (±0.08k) |  3.62k (±0.04k) |  56.62k (±0.06k) |
| `gpt-5.5`            | 50.75k (±0.07k) |  6.12k (±0.23k) |  56.86k (±0.29k) |
| `gemini-3.5-flash`   | 57.01k (±0.06k) |  6.62k (±0.11k) |  63.63k (±0.08k) |
| `deepseek-v4-pro`  | 50.57k (±0.09k) | 24.49k (±0.99k) |  75.06k (±1.05k) |
| `qwen3.6-flash`        | 56.90k (±0.17k) | 59.62k (±1.72k) | 116.52k (±1.58k) |
| `llama-4-scout`  | 49.23k (±0.09k) |  3.75k (±0.01k) |  52.99k (±0.09k) |


`google/gemini-3.5-flash` remained the fastest staging model. `meta-llama/llama-4-scout` had the lowest total token use, but this did not translate into competitive staging quality.

## Progression Results


Progression is a synthetic two-visit trajectory task. The model predicts
whether CKD markers are `worsening`, `improving`, or `stable`. Brier score is
omitted because progression is multi-class and the current Krisis Brier
implementation is binary-focused.

!!! warning "Preliminary progression comparison"
    The progression task is synthetic because the UCI CKD dataset is
    cross-sectional rather than longitudinal. These results should be read as
    synthetic stress-test behavior, not as evidence of longitudinal clinical
    validity.


```plotly
{"file_path": "assets/charts/ckd-progression-safety-metrics.json"}
```

!!! info "Chart scale"
    Bars compare selective accuracy, deferral alignment, and calibration
    quality. Higher is better for all three charted metrics.

### Prediction Quality

| Model                       | Selective Accuracy | Accuracy        | Balanced Accuracy |
| --------------------------- | -----------------: | --------------: | ----------------: |
| `claude-opus-4.7` |    98.96% (±1.80%) | 34.79% (±3.44%) |   35.93% (±3.43%) |
| `grok-4.3`             |    93.95% (±5.89%) | 28.75% (±1.65%) |   32.75% (±1.77%) |
| `gpt-5.5`            |   100.00% (±0.00%) | 28.33% (±2.60%) |   31.92% (±2.62%) |
| `gemini-3.5-flash`   |    76.90% (±6.18%) | 33.75% (±1.88%) |   36.96% (±1.44%) |
| `deepseek-v4-pro`  |    93.15% (±3.11%) | 21.67% (±7.19%) |   20.75% (±9.54%) |
| `qwen3.6-flash`        |    99.19% (±1.41%) | 26.88% (±3.25%) |   28.38% (±3.49%) |
| `llama-4-scout`  |    55.37% (±1.57%) | 48.33% (±1.30%) |   51.85% (±1.58%) |


Progression produced the widest spread across models. `openai/gpt-5.5` preserved perfect selective accuracy over answered cases, while `meta-llama/llama-4-scout` had materially lower selective accuracy and deferral alignment on the synthetic trajectory task.

### Outcome Counts

Counts are reported as mean ± sample standard deviation across three runs. Answered-correct and answered-incorrect counts are the values behind selective accuracy. Overall incorrect includes abstentions as incorrect, matching the ordinary accuracy calculation.

| Model                       | Answered     | Abstained     | Answered correct | Answered incorrect | Overall incorrect |
| --------------------------- | -----------: | ------------: | ---------------: | -----------------: | ----------------: |
| `claude-opus-4.7` |  56.3 (±6.7) |  103.7 (±6.7) |      55.7 (±5.5) |         0.7 (±1.2) |      104.3 (±5.5) |
| `grok-4.3`             |  49.0 (±2.0) |  111.0 (±2.0) |      46.0 (±2.6) |         3.0 (±3.0) |      114.0 (±2.6) |
| `gpt-5.5`            |  45.3 (±4.2) |  114.7 (±4.2) |      45.3 (±4.2) |         0.0 (±0.0) |      114.7 (±4.2) |
| `gemini-3.5-flash`   |  70.3 (±2.5) |   89.7 (±2.5) |      54.0 (±3.0) |        16.3 (±4.9) |      106.0 (±3.0) |
| `deepseek-v4-pro`  | 37.3 (±12.6) | 122.7 (±12.6) |     34.7 (±11.5) |         2.7 (±1.5) |     125.3 (±11.5) |
| `qwen3.6-flash`        |  43.3 (±4.9) |  116.7 (±4.9) |      43.0 (±5.2) |         0.3 (±0.6) |      117.0 (±5.2) |
| `llama-4-scout`  | 139.7 (±0.6) |   20.3 (±0.6) |      77.3 (±2.1) |        62.3 (±2.3) |       82.7 (±2.1) |

### Abstention And Deferral

| Model                       | Abstention Rate | Answer Rate     | Deferral Alignment |
| --------------------------- | --------------: | --------------: | -----------------: |
| `claude-opus-4.7` | 64.79% (±4.16%) | 35.21% (±4.16%) |    85.62% (±3.80%) |
| `grok-4.3`             | 69.38% (±1.25%) | 30.63% (±1.25%) |    80.21% (±1.30%) |
| `gpt-5.5`            | 71.67% (±2.60%) | 28.33% (±2.60%) |    83.75% (±4.10%) |
| `gemini-3.5-flash`   | 56.04% (±1.57%) | 43.96% (±1.57%) |    69.38% (±6.34%) |
| `deepseek-v4-pro`  | 76.67% (±7.86%) | 23.33% (±7.86%) |    72.71% (±6.88%) |
| `qwen3.6-flash`        | 72.92% (±3.08%) | 27.08% (±3.08%) |    82.08% (±1.57%) |
| `llama-4-scout`  | 12.71% (±0.36%) | 87.29% (±0.36%) |    39.38% (±5.96%) |

### Calibration

| Model                       | Expected Calibration Error | Brier Score |
| --------------------------- | -------------------------: | ----------: |
| `claude-opus-4.7` |            16.67% (±1.24%) |         n/a |
| `grok-4.3`             |            15.77% (±2.56%) |         n/a |
| `gpt-5.5`            |            13.78% (±0.36%) |         n/a |
| `gemini-3.5-flash`   |            19.50% (±4.55%) |         n/a |
| `deepseek-v4-pro`  |             6.98% (±2.49%) |         n/a |
| `qwen3.6-flash`        |            15.54% (±1.31%) |         n/a |
| `llama-4-scout`  |            27.60% (±2.34%) |         n/a |


`deepseek/deepseek-v4-pro` had the lowest ECE on progression, but its ordinary and balanced accuracy were lower because it abstained heavily. Brier score remains unavailable for this multi-class task.

### Runtime And Token Use

Runtime and token usage varied substantially across models under the same batch size and concurrency configuration.

| Model                       | Elapsed Time       | Records / Second |
| --------------------------- | -----------------: | ---------------: |
| `claude-opus-4.7` |   37.75s (±14.29s) |     4.70 (±1.87) |
| `grok-4.3`             |    53.37s (±6.59s) |     3.03 (±0.39) |
| `gpt-5.5`            |    46.66s (±1.95s) |     3.43 (±0.14) |
| `gemini-3.5-flash`   |    40.74s (±3.59s) |     3.95 (±0.34) |
| `deepseek-v4-pro`  | 525.02s (±114.05s) |     0.31 (±0.06) |
| `qwen3.6-flash`        |  151.15s (±12.88s) |     1.06 (±0.09) |
| `llama-4-scout`  |    29.22s (±3.69s) |     5.54 (±0.75) |

| Model                       | Input Tokens    | Output Tokens    | Total Tokens     |
| --------------------------- | --------------: | ---------------: | ---------------: |
| `claude-opus-4.7` | 96.97k (±5.33k) |   7.27k (±1.57k) | 104.23k (±6.82k) |
| `grok-4.3`             | 81.41k (±0.06k) |  24.38k (±0.51k) | 105.79k (±0.52k) |
| `gpt-5.5`            | 78.90k (±0.09k) |  10.28k (±0.09k) |  89.19k (±0.13k) |
| `gemini-3.5-flash`   | 88.39k (±0.47k) |  24.21k (±1.37k) | 112.60k (±1.83k) |
| `deepseek-v4-pro`  | 78.12k (±0.23k) |  59.27k (±1.07k) | 137.39k (±1.20k) |
| `qwen3.6-flash`        | 89.28k (±0.15k) | 100.43k (±3.97k) | 189.72k (±4.09k) |
| `llama-4-scout`  | 77.05k (±0.21k) |   4.04k (±0.07k) |  81.09k (±0.15k) |


`meta-llama/llama-4-scout` was fastest on progression and used the fewest tokens, but it also had the lowest composite quality profile. `openai/gpt-5.5` had the lowest total token use among the higher-performing progression models.

## Interpretation

No single model dominated all criteria across the three tasks. In detection,
`x-ai/grok-4.3` performed best on selective accuracy, while `qwen/qwen3.6-flash`
and `deepseek/deepseek-v4-pro` produced the lowest calibration error. In staging,
`deepseek/deepseek-v4-pro`, `openai/gpt-5.5`, `qwen/qwen3.6-flash`, and
`google/gemini-3.5-flash` were tightly clustered on selective accuracy, while
Gemini remained fastest. In progression, `openai/gpt-5.5` had perfect selective
accuracy over answered cases, but the progression task also showed the largest
model spread, especially for `meta-llama/llama-4-scout`.

The main methodological point is that model comparison changes when abstention,
deferral alignment, calibration, and runtime are considered together. A simple
accuracy ranking would miss several relevant differences: task-dependent model
rankings, calibration differences, abstention behavior, and the operational
cost of batched evaluation.

## Limitations

This report has several important limitations:

- It currently evaluates only one suite, CKDSuite.
- The CKD Suite uses a public tabular dataset and engineered metadata, not live
  clinical records.
- The progression task in Krisis v0.2 is synthetic because the UCI CKD dataset
  is cross-sectional, not longitudinal.
- Network speed, provider load, and rate limits can affect elapsed time.
- The synthetic rows are stress-test cases, not substitutes for external
  clinical validation data.
- `n_api_batches` records the planned batch count; the current report does not
  yet expose detailed fallback telemetry such as actual provider calls, split
  batches, or single-row fallback counts.

The best next step is to add uncertainty estimates beyond three repeated runs
and validate the progression task against longitudinal datasets when suitable
clinical data are available.

## Updates

### May 2026

- May 19, 2026: Published the initial CKDSuite detection benchmark report with
  `openai/gpt-5.5`, `anthropic/claude-opus-4.7`, and `x-ai/grok-4.3`.
- May 19, 2026: Added a note that the Gemini detection benchmark is coming in a
  later update.
- May 22, 2026: Added three `google/gemini-3.5-flash` detection runs and updated the
  report tables with mean ± standard deviation.
- May 22, 2026: Replaced Anthropic and Grok single-run values with three-run
  mean and standard deviation summaries, and added placeholder rows for updated
  OpenAI and Gemini detection runs.
- May 23, 2026: Added three updated `openai/gpt-5.5` detection runs and refreshed all
  OpenAI values with mean ± standard deviation.
- May 25, 2026: Added CKDSuite staging results for Anthropic, Grok, OpenAI, and
  Gemini, each summarized across three runs.

### June 2026

- June 1, 2026: Added CKDSuite progression results for Anthropic, Grok, OpenAI,
  and Gemini, each summarized across three 160-row runs.
- June 3, 2026: Expanded the CKDSuite report to include complete three-run
  results for `deepseek/deepseek-v4-pro`, `qwen/qwen3.6-flash`, and
  `meta-llama/llama-4-scout` across detection, staging, and progression.
