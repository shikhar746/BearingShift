"""Tidy index of the Paderborn (KAt) bearing dataset and raw-signal loading.

One row per recording. Labels come from configs/paderborn_bearings.csv (transcribed
from the dataset documentation), never from guessing at file names. File names are
only used for the documented naming convention: <condition>_<bearing>_<repetition>.mat
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

# The four operating conditions of the Paderborn test rig.
# Code: N = speed (x100 rpm), M = load torque (x0.1 Nm), F = radial force (x100 N).
CONDITIONS = {
    "N15_M07_F10": {"rpm": 1500, "torque_nm": 0.7, "radial_force_n": 1000},
    "N09_M07_F10": {"rpm": 900, "torque_nm": 0.7, "radial_force_n": 1000},
    "N15_M01_F10": {"rpm": 1500, "torque_nm": 0.1, "radial_force_n": 1000},
    "N15_M07_F04": {"rpm": 1500, "torque_nm": 0.7, "radial_force_n": 400},
}

FILENAME_RE = re.compile(
    r"^(?P<condition>N\d{2}_M\d{2}_F\d{2})_(?P<bearing_id>K[A-Z0-9]\d{2})_(?P<repetition>\d+)\.mat$"
)

# How a bearing's damage maps to the 3-class problem. Combined (inner + outer)
# damage has no single label, so it maps to NA and is excluded by default.
LABEL_FROM_LOCATION = {"none": "healthy", "inner": "inner", "outer": "outer", "combined": pd.NA}


def load_bearing_table(path):
    """Read the bearing metadata table and add the 3-class `label` column."""
    table = pd.read_csv(path, dtype=str)
    unknown = set(table["fault_location"]) - set(LABEL_FROM_LOCATION)
    if unknown:
        raise ValueError(f"Unknown fault_location values: {unknown}")
    table["label"] = table["fault_location"].map(LABEL_FROM_LOCATION)
    return table


def parse_filename(name):
    """Split a recording file name into condition, bearing_id and repetition."""
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"File name does not follow the Paderborn convention: {name}")
    parts = match.groupdict()
    parts["repetition"] = int(parts["repetition"])
    return parts


def build_index(raw_dir, bearing_table):
    """Scan raw_dir/<bearing_id>/*.mat and return one row per recording."""
    rows = []
    for path in sorted(Path(raw_dir).rglob("*.mat")):
        parts = parse_filename(path.name)
        if path.parent.name != parts["bearing_id"]:
            raise ValueError(f"{path} sits in folder {path.parent.name}, expected {parts['bearing_id']}")
        if parts["condition"] not in CONDITIONS:
            raise ValueError(f"Undocumented operating condition in {path.name}")
        rows.append({"recording_id": path.stem, **parts, **CONDITIONS[parts["condition"]], "path": str(path)})
    if not rows:
        raise FileNotFoundError(f"No .mat files found under {raw_dir}")

    index = pd.DataFrame(rows)
    missing = set(index["bearing_id"]) - set(bearing_table["bearing_id"])
    if missing:
        raise ValueError(f"Bearings not in the documented table: {sorted(missing)}")
    index = index.merge(bearing_table[["bearing_id", "damage_type", "fault_location", "label"]], on="bearing_id")
    columns = ["recording_id", "bearing_id", "condition", "repetition", "rpm", "torque_nm",
               "radial_force_n", "damage_type", "fault_location", "label", "path"]
    return index[columns].sort_values("recording_id").reset_index(drop=True)


def load_vibration(path, channel="vibration_1"):
    """Return one channel of a Paderborn .mat file as a 1-D float array.

    Each file holds one MATLAB struct (named after the file) whose field Y is a list
    of measured channels, each with a Name and a Data field.
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected one struct in {path}, found {keys}")
    for ch in np.atleast_1d(mat[keys[0]].Y):
        if str(ch.Name) == channel:
            return np.asarray(ch.Data, dtype=np.float64).ravel()
    raise KeyError(f"Channel {channel!r} not found in {path}")
