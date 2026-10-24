"""Build the tidy recording index (one row per .mat file) and print a sanity summary.

Usage: python scripts/build_index.py --config configs/default.yaml
"""
import argparse
from pathlib import Path

import yaml

from bearingshift.data import build_index, load_bearing_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(parser.parse_args().config))

    index = build_index(cfg["paths"]["raw_dir"], load_bearing_table(cfg["dataset"]["bearing_table"]))
    out = Path(cfg["paths"]["index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out, index=False)

    print(f"Wrote {len(index)} recordings from {index['bearing_id'].nunique()} bearings to {out}")
    print(index.pivot_table(index="bearing_id", columns="condition", values="recording_id",
                            aggfunc="count", fill_value=0).to_string())
    # The documentation says 20 recordings per bearing per condition: flag anything else.
    counts = index.groupby(["bearing_id", "condition"]).size()
    expected = cfg["dataset"].get("recordings_per_condition")
    if expected and (counts != expected).any():
        print(f"WARNING: some bearing/condition pairs do not have {expected} recordings:")
        print(counts[counts != expected].to_string())


if __name__ == "__main__":
    main()
