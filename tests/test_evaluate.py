import numpy as np
import pandas as pd
import pytest

from bearingshift.evaluate import classification_metrics, summarise, to_recording_level


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, np.eye(3)[y])
    assert m["macro_auroc"] == 1 and m["macro_f1"] == 1 and m["recall_inner"] == 1


def test_constant_predictions_give_chance_auroc():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, np.full((6, 3), 1 / 3))
    assert m["macro_auroc"] == pytest.approx(0.5)


def test_recording_level_averages_windows():
    y, p = to_recording_level(["r1", "r1", "r2"], [2, 2, 0],
                              np.array([[0.2, 0.2, 0.6], [0.4, 0.2, 0.4], [0.9, 0.05, 0.05]]))
    np.testing.assert_array_equal(y, [2, 0])
    np.testing.assert_allclose(p[0], [0.3, 0.2, 0.5])


def test_summary_reports_mean_and_std_over_splits():
    per_split = pd.DataFrame({"split_type": "bearing", "model": "m", "level": "window",
                              "macro_auroc": [0.6, 0.8], "macro_f1": [0.5, 0.7],
                              "recall_healthy": 1.0, "recall_inner": 0.0, "recall_outer": 0.5})
    row = summarise(per_split).iloc[0]
    assert row["macro_f1_mean"] == pytest.approx(0.6)
    assert row["macro_f1_std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert row["n_splits"] == 2
