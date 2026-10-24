"""The four rungs of the leakage ladder.

  a. random_window   - windows shuffled at random (leaky: windows of the same recording
                       land in both train and test)
  b. recording       - whole recordings go to one side (bearings still in both sides)
  c. condition       - train on 3 operating conditions, test on the 4th (bearings still
                       in both sides, just recorded under a different condition)
  d. bearing         - no physical bearing in both train and test (the honest one)

Rungs b-d split the RECORDING index; windowing happens afterwards. Rung a is the only one
that splits windows, because that is exactly the shortcut we want to measure.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SPLIT_TYPES = ["random_window", "recording", "condition", "bearing"]


@dataclass
class Split:
    split_type: str
    repeat: int
    level: str        # "window": train/test are row positions in the window table
                      # "recording": train/test are recording_ids
    train: np.ndarray
    test: np.ndarray


def random_window_splits(windows, n_repeats, test_size, seed):
    """Stratified random split of individual windows (the leaky baseline)."""
    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
    return [Split("random_window", r, "window", tr, te)
            for r, (tr, te) in enumerate(sss.split(windows, windows["label"]))]


def recording_splits(recordings, n_repeats, test_size, seed):
    """Stratified random split of whole recordings."""
    ids = recordings["recording_id"].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
    return [Split("recording", r, "recording", ids[tr], ids[te])
            for r, (tr, te) in enumerate(sss.split(recordings, recordings["label"]))]


def condition_splits(recordings):
    """Leave-one-condition-out: 4 splits, one per operating condition."""
    splits = []
    for r, cond in enumerate(sorted(recordings["condition"].unique())):
        is_test = (recordings["condition"] == cond).to_numpy()
        ids = recordings["recording_id"].to_numpy()
        splits.append(Split("condition", r, "recording", ids[~is_test], ids[is_test]))
    return splits


def bearing_splits(recordings, n_repeats, test_fraction, seed):
    """Repeated bearing-wise split, stratified by class.

    For each repeat and each class, a random `test_fraction` of that class's bearings
    (at least 1, and at least 1 left for training) goes to test. Every test set therefore
    contains all classes, which macro-AUROC needs.
    """
    bearing_label = recordings.groupby("bearing_id")["label"].agg(lambda s: s.unique())
    if bearing_label.map(len).max() > 1:
        raise ValueError("A bearing has more than one label")
    bearing_label = bearing_label.map(lambda a: a[0])

    splits = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        test_bearings = []
        for _, ids in bearing_label.groupby(bearing_label):
            ids = np.sort(ids.index.to_numpy())
            k = min(max(1, round(test_fraction * len(ids))), len(ids) - 1)
            test_bearings.extend(rng.choice(ids, size=k, replace=False))
        is_test = recordings["bearing_id"].isin(test_bearings).to_numpy()
        ids = recordings["recording_id"].to_numpy()
        splits.append(Split("bearing", r, "recording", ids[~is_test], ids[is_test]))
    return splits


def make_splits(split_type, recordings, windows, cfg, seed):
    """Dispatch by name, with parameters from the config's `splits` section."""
    params = cfg["splits"].get(split_type) or {}
    if split_type == "random_window":
        return random_window_splits(windows, params["n_repeats"], params["test_size"], seed)
    if split_type == "recording":
        return recording_splits(recordings, params["n_repeats"], params["test_size"], seed)
    if split_type == "condition":
        return condition_splits(recordings)
    if split_type == "bearing":
        return bearing_splits(recordings, params["n_repeats"], params["test_fraction"], seed)
    raise ValueError(f"Unknown split type: {split_type}")


def shared_groups(recordings, split, column):
    """Values of `column` (e.g. bearing_id) that appear in BOTH train and test."""
    by_id = recordings.set_index("recording_id")[column]
    return set(by_id.loc[split.train]) & set(by_id.loc[split.test])
