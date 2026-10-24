import numpy as np
import pandas as pd

from bearingshift.features import TIME_FEATURES
from bearingshift.models import PhysicsRule, fit_model


def toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = pd.DataFrame(rng.normal(size=(n, len(TIME_FEATURES))), columns=TIME_FEATURES)
    for h in (1, 2, 3):
        X[f"env_bpfo_h{h}"] = rng.normal(1, 0.1, n) + 3 * (y == 2)
        X[f"env_bpfi_h{h}"] = rng.normal(1, 0.1, n) + 3 * (y == 1)
    meta = pd.DataFrame({"bearing_id": rng.integers(0, 12, n).astype(str)})
    return X, y, meta


def test_physics_rule_learns_separable_thresholds():
    X, y, _ = toy_data()
    rule = PhysicsRule().fit(X, y)
    assert (rule.predict(X) == y).mean() > 0.95
    assert rule.predict_proba(X).shape == (len(y), 3)


def test_scaler_statistics_come_from_training_rows_only():
    X, y, meta = toy_data()
    train = np.arange(len(y)) < 200
    X.loc[~train, TIME_FEATURES] += 100  # a test shift the scaler must NOT see
    model = fit_model("logistic_time", X[train], y[train], meta[train], "bearing", cfg={}, seed=0)
    scaler = model.model.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, X.loc[train, TIME_FEATURES].mean().to_numpy())


def test_lightgbm_tuning_runs_with_bearing_groups():
    X, y, meta = toy_data()
    cfg = {"lightgbm": {"inner_folds": 3, "param_grid": {"num_leaves": [7], "n_estimators": [20]}}}
    search = fit_model("lightgbm", X, y, meta, "bearing", cfg, seed=0)
    assert search.predict_proba(X).shape == (len(y), 3)
