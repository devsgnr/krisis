"""CKD data pipeline tests."""

from __future__ import annotations

import pandas as pd
import pytest

from krisis.data.base import FeatureSet, SuiteConfig, Task
from krisis.data.ckd.engineer import CKDFeatureEngineer
from krisis.data.ckd.preprocess import FINAL_COLUMN_ORDER, CKDPreprocessor
from krisis.data.ckd.suite import DEFAULT_DATA_PATH, CKDSuite
from krisis.data.ckd.validate import RAW_COLUMN_ORDER, validate_ckd_csv


def _has_nan(value: object) -> bool:
    if isinstance(value, dict):
        return any(_has_nan(v) for v in value.values())
    return isinstance(value, float) and value != value


def test_ckd_preprocessor_keeps_unscaled_clinical_frame() -> None:
    raw = pd.read_csv(DEFAULT_DATA_PATH)
    preprocessor = CKDPreprocessor(feature_set=FeatureSet.REDUCED, seed=7)

    scaled = preprocessor.fit_transform(raw)
    clinical = preprocessor.get_imputed_dataframe()

    assert list(scaled.columns) == [*FINAL_COLUMN_ORDER, "class"]
    assert clinical["age"].max() > 1.0
    assert clinical["sc"].max() > 1.0

    engineered = CKDFeatureEngineer(seed=7).fit_transform(clinical)
    assert engineered["egfr"].notna().all()
    assert (engineered["egfr"] > 0).all()
    assert set(engineered["ckd_stage"]).issubset({1, 2, 3, 4, 5})


def test_ckd_csv_validation_reorders_canonical_columns() -> None:
    raw = pd.read_csv(DEFAULT_DATA_PATH)
    shuffled = raw[list(reversed(raw.columns))]

    result = validate_ckd_csv(shuffled, data_path="shuffled.csv")

    assert list(result.dataframe.columns) == RAW_COLUMN_ORDER
    assert result.n_rows == len(raw)
    assert result.n_missing_values > 0


def test_ckd_suite_rejects_missing_required_csv_columns(tmp_path) -> None:
    raw = pd.read_csv(DEFAULT_DATA_PATH).drop(columns=["sc"])
    bad_path = tmp_path / "missing_sc.csv"
    raw.to_csv(bad_path, index=False)
    suite = CKDSuite(data_path=str(bad_path))

    with pytest.raises(ValueError, match="missing required UCI CKD columns"):
        suite.load()


def test_ckd_suite_rejects_invalid_categorical_values(tmp_path) -> None:
    raw = pd.read_csv(DEFAULT_DATA_PATH)
    raw.loc[0, "htn"] = "sometimes"
    bad_path = tmp_path / "bad_htn.csv"
    raw.to_csv(bad_path, index=False)
    suite = CKDSuite(data_path=str(bad_path))

    with pytest.raises(ValueError, match="column 'htn' contains unsupported values"):
        suite.load()


def test_ckd_suite_rejects_duplicate_ids(tmp_path) -> None:
    raw = pd.read_csv(DEFAULT_DATA_PATH)
    raw.loc[1, "id"] = raw.loc[0, "id"]
    bad_path = tmp_path / "duplicate_id.csv"
    raw.to_csv(bad_path, index=False)
    suite = CKDSuite(data_path=str(bad_path))

    with pytest.raises(ValueError, match="id.*unique"):
        suite.load()


def test_ckd_suite_reduced_load_uses_clinical_units() -> None:
    suite = CKDSuite(
        config=SuiteConfig(
            features=FeatureSet.REDUCED,
            task=Task.DETECTION,
            seed=11,
            n_synthetic=4,
            test_size=0.2,
        ),
    )

    records = suite.load()

    assert len(records) == 84
    assert set(records[0].features) == set(FINAL_COLUMN_ORDER)
    assert max(record.features["sc"] for record in records) > 1.0
    assert all("egfr" in record.metadata for record in records)
    assert all("should_abstain" in record.metadata for record in records)
    assert any(record.metadata["should_abstain"] for record in records)
    assert suite.describe()["n_synthetic_records"] == 4
    assert suite.describe()["n_should_abstain_records"] > 0


def test_ckd_suite_progression_builds_two_visit_records() -> None:
    suite = CKDSuite(
        config=SuiteConfig(
            features=FeatureSet.REDUCED,
            task=Task.PROGRESSION,
            seed=42,
            n_synthetic=10,
            test_size=0.2,
        ),
    )

    records = suite.load()
    labels = {record.label for record in records}

    assert len(records) == 90
    assert labels.issubset({"stable", "worsening", "improving"})
    assert {"stable", "worsening", "improving"}.issubset(labels)
    assert set(records[0].features) == {"trajectory_months", "baseline", "current"}
    assert set(records[0].features["baseline"]) == set(FINAL_COLUMN_ORDER)
    assert set(records[0].features["current"]) == set(FINAL_COLUMN_ORDER)
    assert any(record.metadata.get("progression_ambiguous") for record in records)
    assert any(
        record.metadata.get("deferral_reason") == "ambiguous_progression_trajectory"
        for record in records
    )
    assert suite.describe()["progression_distribution"]


def test_ckd_suite_staging_exposes_egfr_for_stage_assignment() -> None:
    suite = CKDSuite(
        config=SuiteConfig(
            features=FeatureSet.REDUCED,
            task=Task.STAGING,
            seed=42,
            n_synthetic=10,
            test_size=0.2,
        ),
    )

    records = suite.load()

    assert len(records) == 90
    assert "egfr" in records[0].features
    assert "egfr_threshold_margin" in records[0].features
    assert "nearest_egfr_stage_threshold" in records[0].features
    assert "egfr" in records[0].metadata
    assert records[0].label in {1, 2, 3, 4, 5}


def test_ckd_suite_full_feature_set_with_synthetic_has_no_missing_features() -> None:
    suite = CKDSuite(
        config=SuiteConfig(
            features=FeatureSet.FULL,
            task=Task.STAGING,
            seed=42,
            n_synthetic=4,
            test_size=0.2,
        ),
    )

    records = suite.load()

    assert len(records) == 84
    assert "age" in records[0].features
    assert "wbcc" in records[0].features
    assert "egfr" in records[0].features
    assert not any(_has_nan(record.features) for record in records)
