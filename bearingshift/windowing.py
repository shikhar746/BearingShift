"""Cut recordings into windows and compute per-window features.

Order of operations in the pipeline: split recordings -> window -> features -> fit.
A window's features depend only on that window's samples (nothing is learned here),
so they are cached per recording. Anything that IS learned from data (scalers,
thresholds, models, hyperparameters) is fitted after the split on training rows only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bearingshift.data import load_vibration
from bearingshift.features import window_features

META_COLUMNS = ["recording_id", "window_idx", "bearing_id", "condition", "rpm", "torque_nm",
                "radial_force_n", "damage_type", "label"]


def window_signal(x, fs, window_s, overlap=0.0):
    """Return an (n_windows, window_length) array. A leftover tail shorter than a window is dropped."""
    length = int(round(window_s * fs))
    hop = int(round(length * (1 - overlap)))
    if hop <= 0:
        raise ValueError("overlap must be < 1")
    if len(x) < length:
        return np.empty((0, length))
    starts = range(0, len(x) - length + 1, hop)
    return np.stack([x[s:s + length] for s in starts])


def recording_features(row, cfg, orders):
    """Feature rows (one per window) for one recording of the index."""
    fs = cfg["dataset"]["fs_hz"]
    x = load_vibration(row["path"], cfg["dataset"]["vibration_channel"])
    windows = window_signal(x, fs, cfg["windowing"]["window_s"], cfg["windowing"]["overlap"])
    shaft_hz = row["rpm"] / 60.0  # nominal speed of the documented operating condition
    rows = []
    for i, w in enumerate(windows):
        meta = {c: row[c] for c in META_COLUMNS if c in row}
        rows.append({**meta, "window_idx": i, **window_features(w, fs, shaft_hz, orders, cfg["features"])})
    return rows


def feature_table(recordings, cfg, orders):
    """Per-window features for the given recordings, cached on disk by feature settings."""
    settings = {k: cfg[k] for k in ("windowing", "features", "geometry")} | {"fs": cfg["dataset"]["fs_hz"]}
    key = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    cache = Path(cfg["paths"]["feature_cache_dir"]) / f"window_features_{key}.csv"

    cached = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["recording_id"])
    todo = recordings[~recordings["recording_id"].isin(cached["recording_id"])]
    if len(todo):
        print(f"Computing features for {len(todo)} recordings (cache: {cache})")
        new_rows = [r for _, row in todo.iterrows() for r in recording_features(row, cfg, orders)]
        cached = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache, index=False)

    table = cached[cached["recording_id"].isin(recordings["recording_id"])]
    return table.sort_values(["recording_id", "window_idx"]).reset_index(drop=True)


def feature_columns(table):
    return [c for c in table.columns if c not in META_COLUMNS]
