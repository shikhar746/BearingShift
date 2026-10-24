import itertools

import pandas as pd
import pytest

from bearingshift.data import CONDITIONS, load_bearing_table
from bearingshift.splits import (bearing_splits, condition_splits, random_window_splits,
                                 recording_splits, shared_groups)


@pytest.fixture
def recordings():
    """Index with the real Paderborn structure (29 labelled bearings x 4 conditions x 20 reps)."""
    table = load_bearing_table("configs/paderborn_bearings.csv").dropna(subset=["label"])
    rows = [{"recording_id": f"{c}_{b}_{r}", "bearing_id": b, "condition": c, "label": lab}
            for (b, lab), c, r in itertools.product(zip(table["bearing_id"], table["label"]), CONDITIONS, range(1, 21))]
    return pd.DataFrame(rows)


def test_bearing_split_never_shares_a_bearing(recordings):
    splits = bearing_splits(recordings, n_repeats=20, test_fraction=0.25, seed=0)
    assert len(splits) >= 20
    for split in splits:
        assert shared_groups(recordings, split, "bearing_id") == set()


def test_bearing_split_test_set_has_every_class(recordings):
    by_id = recordings.set_index("recording_id")["label"]
    for split in bearing_splits(recordings, n_repeats=20, test_fraction=0.25, seed=0):
        assert set(by_id.loc[split.test]) == {"healthy", "inner", "outer"}
        assert set(by_id.loc[split.train]) == {"healthy", "inner", "outer"}


def test_bearing_splits_differ_between_repeats(recordings):
    splits = bearing_splits(recordings, n_repeats=20, test_fraction=0.25, seed=0)
    assert len({tuple(sorted(s.test)) for s in splits}) > 1


def test_condition_split_holds_out_one_condition(recordings):
    by_id = recordings.set_index("recording_id")["condition"]
    splits = condition_splits(recordings)
    assert len(splits) == 4
    for split in splits:
        assert by_id.loc[split.test].nunique() == 1
        assert shared_groups(recordings, split, "condition") == set()


def test_recording_split_keeps_recordings_whole(recordings):
    for split in recording_splits(recordings, n_repeats=5, test_size=0.25, seed=0):
        assert set(split.train).isdisjoint(split.test)
        assert len(split.train) + len(split.test) == len(recordings)


def test_random_window_split_is_leaky_on_purpose(recordings):
    windows = recordings.loc[recordings.index.repeat(4)].reset_index(drop=True)
    split = random_window_splits(windows, n_repeats=1, test_size=0.25, seed=0)[0]
    shared = set(windows.loc[split.train, "recording_id"]) & set(windows.loc[split.test, "recording_id"])
    assert len(shared) > 0  # the shortcut this project measures
