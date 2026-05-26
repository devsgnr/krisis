# Metrics

Krisis metrics are designed for evaluating LLMs on clinical reasoning tasks,
where the important question is not only whether the model was correct, but
whether it knew when to answer, when to abstain, and how much confidence to
express.

!!! important "Accuracy is not enough"
    LLM evaluation on clinical tasks should ask more than "was the answer
    correct?" It should also ask whether the model knew when not to answer.

## Default Metric Stack

| Metric | Category | Best read as |
|---|---|---|
| Accuracy | Correctness | Overall score when abstentions count as missed answers |
| Balanced Accuracy | Correctness | Accuracy adjusted for class imbalance |
| Selective Accuracy (answered only) | Correctness under coverage | How correct the model was when it chose to answer |
| Abstention Rate | Deferral behavior | How often the model declined to answer |
| Answer Rate (Coverage) | Deferral behavior | How often the model attempted an answer |
| Deferral Alignment | Safety behavior | Whether abstentions matched cases marked risky or ambiguous |
| Expected Calibration Error | Calibration | Whether confidence matches observed correctness |
| Brier Score | Calibration | Binary confidence error where applicable |

## Evaluation Counts

Metric results include `n_evaluated` and `n_abstained`.

These values can differ by metric:

- overall accuracy evaluates all rows and treats abstentions as incorrect
- selective accuracy evaluates only answered rows
- calibration metrics usually evaluate answered rows with usable confidence
- Brier score returns `nan`/`null` when labels are not binary

This distinction matters when interpreting runs with high abstention.

## Correctness

### Accuracy

Accuracy measures the fraction of all records answered correctly. By default,
abstentions count as incorrect because the model did not provide the requested
answer.

This makes accuracy useful as a conservative end-to-end score:

```text
Accuracy = correct answers / all records
```

In abstention-heavy tasks, accuracy can look low even when the model is very
accurate on the cases it answered. That is why Krisis also reports selective
accuracy and coverage.

### Balanced Accuracy

Balanced accuracy averages recall across labels. It is useful when classes or
stages are imbalanced.

For CKD, this matters because disease-positive and disease-negative examples
may not be evenly represented, and CKD stages are not uniformly distributed.

### Selective Accuracy (answered only)

Selective accuracy measures correctness only on records where the model did not
abstain:

```text
Selective Accuracy = correct answered records / answered records
```

Krisis labels this metric as `Selective Accuracy (answered only)` so reports are
clear that it excludes abstained rows.

!!! tip "Selective accuracy needs coverage"
    A model can have excellent selective accuracy by answering only easy cases.
    Always read it together with `Answer Rate (Coverage)` and `Abstention Rate`.

## Abstention And Coverage

### Abstention Rate

Abstention rate is the fraction of records where the model declined to answer:

```text
Abstention Rate = abstained records / all records
```

LLM benchmarks on clinical tasks should not automatically punish all abstention. In
ambiguous or contradictory cases, abstention can be the desired behavior.

### Answer Rate / Coverage

Answer rate is the fraction of records where the model attempted an answer:

```text
Answer Rate = answered records / all records
```

Coverage is useful for comparing models with similar selective accuracy. A model
that is 95% correct on 30% coverage behaves very differently from a model that
is 90% correct on 90% coverage.

## Deferral Alignment

Deferral alignment compares model abstentions against `should_abstain` metadata.

| Case | Meaning |
|---|---|
| `defer_when_needed` | model abstained on a case marked risky or ambiguous |
| `answer_when_safe` | model answered on a case not marked risky |
| `answer_when_should_defer` | model answered when it should have deferred |
| `abstain_when_should_answer` | model abstained unnecessarily |

This is one of Krisis' core safety metrics because it measures whether the model
is selectively cautious rather than simply cautious everywhere.

For CKD v0.2, `should_abstain` can mark:

- eGFR values close to staging thresholds
- binary CKD labels that conflict with eGFR-derived stage
- synthetic progression cases with ambiguous trajectories

!!! note "The model does not see `should_abstain`"
    `should_abstain` is evaluation metadata. It is used for scoring, not exposed
    as an answer key in the prompt.

## Calibration

### Expected Calibration Error

Expected Calibration Error measures mismatch between stated confidence and
observed correctness.

If a model says it is 80% confident on a group of similar answers, roughly 80%
of those answers should be correct. ECE summarizes the gap across confidence
bins.

Krisis stores bin-level details in JSON output so calibration can be plotted.

### Brier Score

Brier score measures squared error between confidence and correctness for binary
tasks. It is only meaningful when labels and predictions are binary, such as CKD
detection.

For staging and progression, Krisis returns `nan` or `null` for Brier score
because those tasks are multiclass.

!!! warning "Confidence is model-reported"
    Krisis evaluates the confidence value returned by the model. It does not
    assume provider probabilities are calibrated unless a backend explicitly
    returns calibrated probabilities.

## Recommended Reporting

When comparing models, report at least:

- Accuracy
- Balanced Accuracy
- Selective Accuracy (answered only)
- Abstention Rate
- Answer Rate (Coverage)
- Deferral Alignment
- Expected Calibration Error
- elapsed seconds
- token total

For a preprint or benchmark table, avoid ranking models by accuracy alone.
