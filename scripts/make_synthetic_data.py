"""Write a small synthetic Paderborn-like dataset for the smoke test (NOT real data).

Usage: python scripts/make_synthetic_data.py --config configs/smoke.yaml
"""
import argparse

import yaml

from bearingshift.data import load_bearing_table
from bearingshift.features import BearingGeometry, defect_orders
from bearingshift.synthetic import make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    cfg = yaml.safe_load(open(parser.parse_args().config))
    make_dataset(cfg["paths"]["raw_dir"], load_bearing_table(cfg["dataset"]["bearing_table"]),
                 defect_orders(BearingGeometry(**cfg["geometry"])),
                 n_repetitions=cfg["dataset"]["recordings_per_condition"], fs=cfg["dataset"]["fs_hz"])
    print(f"Synthetic dataset written to {cfg['paths']['raw_dir']}")


if __name__ == "__main__":
    main()
