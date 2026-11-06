import numpy as np
import pandas as pd

from run_calibrated_temporal_research import (
    calculate_psi, expected_calibration_error, psi_categorical, psi_numeric
)


def test_categorical_psi_detects_calendar_distribution_shift():
    reference = pd.Series([0, 0, 0, 1, 1, 2])
    current = pd.Series([2, 2, 2, 2, 1, 1])
    assert psi_categorical(reference, current) > 0.1


def test_numeric_psi_is_zero_for_identical_distribution():
    values = np.arange(100, dtype=float)
    assert abs(psi_numeric(values, values)) < 1e-12


def test_numeric_psi_detects_shift_and_handles_constant_feature():
    reference = np.arange(100, dtype=float)
    shifted = np.arange(100, dtype=float) + 1000
    assert psi_numeric(reference, shifted) > 0.1
    assert psi_numeric(np.ones(20), np.ones(10)) == 0.0
    assert psi_numeric(np.ones(20), np.ones(10) * 5) > 0.1


def test_calendar_fields_use_categorical_psi():
    result = calculate_psi([0, 1, 1, 2], [2, 2, 2, 3], "event_dow")
    assert result["method"] == "categorical"
    assert result["psi"] > 0


def test_ece_is_finite_and_reports_bins():
    ece, bins = expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert np.isfinite(ece)
    assert ece < 0.2
    assert sum(row["count"] for row in bins) == 4
