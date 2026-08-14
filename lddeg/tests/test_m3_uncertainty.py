"""Uncertainty and confidence reporting tests (M3 Parts 5 and 11)."""
import numpy as np
import pytest

from src.config import UncertaintyConfig
from src.uncertainty import (CONFIDENCE_DISCLAIMER, prediction_confidence,
                             uncertainty_summary, uncertainty_table)


def test_confidence_margin_and_entropy_match_their_definitions():
    probabilities = [[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]]
    measures = prediction_confidence(probabilities)
    assert np.allclose(measures["confidence"], [0.7, 0.4])
    assert np.allclose(measures["margin"], [0.5, 0.05])
    expected = -sum(p * np.log(p) for p in (0.7, 0.2, 0.1)) / np.log(3)
    assert measures["entropy"][0] == pytest.approx(expected)


def test_uniform_probability_is_maximum_entropy():
    measures = prediction_confidence([[0.25] * 4])
    assert measures["entropy"][0] == pytest.approx(1.0)
    assert measures["margin"][0] == pytest.approx(0.0)
    assert measures["uncertain"][0]


def test_certain_prediction_is_minimum_entropy():
    measures = prediction_confidence([[1.0, 0.0, 0.0]])
    assert measures["entropy"][0] == pytest.approx(0.0)
    assert not measures["uncertain"][0]


def test_low_margin_is_flagged_even_when_confidence_is_high():
    """A 0.49/0.48 split must not pass just because the top value is large."""
    cfg = UncertaintyConfig(confidence_threshold=0.40, margin_threshold=0.10)
    measures = prediction_confidence([[0.49, 0.48, 0.03]], cfg=cfg)
    assert measures["confidence"][0] > cfg.confidence_threshold
    assert measures["uncertain"][0], "a contested prediction must be flagged"


def test_thresholds_are_configurable():
    probabilities = [[0.65, 0.35]]
    strict = prediction_confidence(
        probabilities, cfg=UncertaintyConfig(confidence_threshold=0.9,
                                             margin_threshold=0.0))
    lenient = prediction_confidence(
        probabilities, cfg=UncertaintyConfig(confidence_threshold=0.5,
                                             margin_threshold=0.0))
    assert strict["uncertain"][0] and not lenient["uncertain"][0]


def test_unpredicted_rows_are_uncertain_never_confident():
    probabilities = np.array([[0.8, 0.2], [np.nan, np.nan]])
    measures = prediction_confidence(probabilities)
    assert measures["uncertain"][1]
    assert np.isnan(measures["confidence"][1])


def test_malformed_probabilities_are_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        prediction_confidence([[0.2, 0.2]])
    with pytest.raises(ValueError, match="at least two classes"):
        prediction_confidence([[1.0]])
    with pytest.raises(ValueError, match="negative"):
        prediction_confidence([[-0.5, 1.5]])


def test_uncertainty_table_reports_every_required_column():
    probabilities = np.array([[0.7, 0.3], [0.5, 0.5]])
    frame = uncertainty_table([1, 2], probabilities, [1, 2], truth=[1, 1])
    for column in ("truth", "prediction", "correct", "probability_1",
                   "probability_2", "confidence", "margin", "entropy",
                   "uncertain"):
        assert column in frame.columns
    assert frame["correct"].tolist() == [True, False]


def test_summary_reports_accuracy_split_by_the_flag():
    probabilities = np.array([[0.95, 0.05], [0.95, 0.05],
                              [0.52, 0.48], [0.52, 0.48]])
    truth = np.array([1, 1, 1, 2])
    predictions = np.array([1, 1, 1, 1])
    summary = uncertainty_summary(probabilities, truth=truth,
                                  predictions=predictions)
    assert summary["n_uncertain"] == 2
    assert summary["accuracy_confident_subset"] == pytest.approx(1.0)
    assert summary["accuracy_uncertain_subset"] == pytest.approx(0.5)
    assert summary["confidence_is_informative"] is True


def test_summary_does_not_hide_an_uninformative_confidence_measure():
    """If flagged predictions are no worse, the summary must say so."""
    probabilities = np.array([[0.95, 0.05], [0.52, 0.48]])
    summary = uncertainty_summary(probabilities, truth=np.array([2, 1]),
                                  predictions=np.array([1, 1]))
    assert summary["confidence_is_informative"] is False


def test_outputs_carry_the_disclaimer():
    measures = prediction_confidence([[0.6, 0.4]])
    assert measures["disclaimer"] == CONFIDENCE_DISCLAIMER
    assert "not certainty" in CONFIDENCE_DISCLAIMER
    assert "not calibrated" in CONFIDENCE_DISCLAIMER
