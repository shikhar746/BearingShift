import numpy as np
import pandas as pd
import pytest

from bearingshift.data import build_index, load_bearing_table, load_vibration, parse_filename
from bearingshift.synthetic import write_recording

TABLE = "configs/paderborn_bearings.csv"


def test_bearing_table_matches_documented_counts():
    table = load_bearing_table(TABLE)
    assert len(table) == 32
    assert table["damage_type"].value_counts().to_dict() == {"healthy": 6, "artificial": 12, "real": 14}
    # Combined-damage bearings have no single 3-class label.
    assert table.loc[table["label"].isna(), "bearing_id"].tolist() == ["KB23", "KB24", "KB27"]


def test_parse_filename():
    assert parse_filename("N09_M07_F10_KA01_17.mat") == {
        "condition": "N09_M07_F10", "bearing_id": "KA01", "repetition": 17}
    with pytest.raises(ValueError):
        parse_filename("KA01_whatever.mat")


def test_build_index_and_load_roundtrip(tmp_path):
    x = np.arange(100, dtype=float)
    write_recording(tmp_path / "KA01" / "N15_M07_F10_KA01_1.mat", x)
    write_recording(tmp_path / "K001" / "N09_M07_F10_K001_2.mat", x)
    index = build_index(tmp_path, load_bearing_table(TABLE))

    assert len(index) == 2
    row = index.set_index("bearing_id").loc["KA01"]
    assert (row["label"], row["damage_type"], row["rpm"]) == ("outer", "artificial", 1500)
    np.testing.assert_allclose(load_vibration(row["path"]), x)


def test_build_index_rejects_file_in_wrong_folder(tmp_path):
    write_recording(tmp_path / "KA03" / "N15_M07_F10_KA01_1.mat", np.zeros(10))
    with pytest.raises(ValueError):
        build_index(tmp_path, load_bearing_table(TABLE))
