"""Baselines and the main model. Labels are ints 0/1/2 = healthy/inner/outer.

Every model is fitted on training rows only. Nothing here sees test data.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bearingshift.features import TIME_FEATURES

HEALTHY, INNER, OUTER = 0, 1, 2


class PhysicsRule(BaseEstimator, ClassifierMixin):
    """Textbook envelope check: a fault if the envelope peak at BPFO (or BPFI) is high.

    Score_o = mean envelope peak-to-background ratio over the BPFO harmonics (same for
    BPFI). Two thresholds are fitted on TRAIN by grid search over train quantiles,
    maximising macro-F1. Prediction: healthy if both scores are below threshold,
    otherwise the race whose (score / threshold) is larger.
    """

    def __init__(self, n_quantiles=19):
        self.n_quantiles = n_quantiles

    def _scores(self, X):
        o = X[[c for c in X.columns if c.startswith("env_bpfo_")]].mean(axis=1).to_numpy()
        i = X[[c for c in X.columns if c.startswith("env_bpfi_")]].mean(axis=1).to_numpy()
        return o, i

    def _ratio_scores(self, X, thr_o, thr_i):
        o, i = self._scores(X)
        ro, ri = o / thr_o, i / thr_i
        # Columns: healthy, inner, outer. argmax reproduces the rule described above.
        scores = np.column_stack([1 / np.maximum(ro, ri), ri, ro])
        return scores / scores.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        o, i = self._scores(X)
        qs = np.linspace(0.05, 0.95, self.n_quantiles)
        best = -1.0
        for thr_o in np.quantile(o, qs):
            for thr_i in np.quantile(i, qs):
                pred = self._ratio_scores(X, thr_o, thr_i).argmax(axis=1)
                score = f1_score(y, pred, average="macro")
                if score > best:
                    best, self.thr_o_, self.thr_i_ = score, thr_o, thr_i
        self.classes_ = np.array([HEALTHY, INNER, OUTER])
        return self

    def predict_proba(self, X):
        return self._ratio_scores(X, self.thr_o_, self.thr_i_)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class ColumnSubset(BaseEstimator, ClassifierMixin):
    """Fit a model on a fixed subset of feature columns."""

    def __init__(self, model, columns):
        self.model = model
        self.columns = columns

    def fit(self, X, y):
        self.model.fit(X[self.columns], y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])


def inner_cv(split_type, seed, n_folds):
    """Inner CV for hyperparameter tuning, grouped like the outer split.

    A leaky outer protocol gets a leaky inner protocol too, which is what a project
    using that protocol would actually do. Returns (splitter, group column or None).
    """
    if split_type == "random_window":
        return StratifiedKFold(n_folds, shuffle=True, random_state=seed), None
    if split_type == "recording":
        return StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed), "recording_id"
    if split_type == "condition":
        return GroupKFold(n_folds), "condition"  # 3 train conditions -> leave-one-condition-out
    if split_type == "bearing":
        return StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed), "bearing_id"
    raise ValueError(split_type)


def fit_model(name, X, y, meta, split_type, cfg, seed):
    """Fit model `name` on training features X, labels y, metadata `meta` (same rows)."""
    if name == "majority":
        return DummyClassifier(strategy="prior").fit(X, y)
    if name == "physics_rule":
        return PhysicsRule().fit(X, y)
    if name == "logistic_time":
        # The scaler lives inside the pipeline, so its mean/std come from training rows only.
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        return ColumnSubset(pipe, TIME_FEATURES).fit(X, y)
    if name == "lightgbm":
        lcfg = cfg["lightgbm"]
        cv, group_col = inner_cv(split_type, seed, lcfg["inner_folds"])
        base = LGBMClassifier(objective="multiclass", class_weight="balanced", random_state=seed,
                              n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
        search = GridSearchCV(base, lcfg["param_grid"], scoring="f1_macro", cv=cv, refit=True, n_jobs=-1)
        groups = meta[group_col].to_numpy() if group_col else None
        return search.fit(X, y, groups=groups)
    raise ValueError(f"Unknown model: {name}")
