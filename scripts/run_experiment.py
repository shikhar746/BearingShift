"""Run the full leakage ladder: every split type x every model, then SHAP.

Usage: python scripts/run_experiment.py --config configs/default.yaml

Writes to the config's results_dir:
  metrics_per_split.csv   one row per split x model x level (window / recording)
  leakage_ladder.csv      mean, std and n_splits of every metric
  leakage_ladder_window.png, leakage_ladder_recording.png
  shap_importance.csv     LightGBM mean |SHAP| per feature over the bearing-wise splits,
                          with its Spearman correlation to speed, torque and radial force
  shap_importance.png
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch

from bearingshift import CLASSES
from bearingshift.config import load_config
from bearingshift.evaluate import (MODEL_COLORS, classification_metrics, plot_ladder, summarise,
                                   to_recording_level)
from bearingshift.features import BearingGeometry, defect_orders, feature_family
from bearingshift.models import fit_model
from bearingshift.splits import make_splits, shared_groups
from bearingshift.windowing import feature_columns, feature_table

CONDITION_VARIABLES = ["rpm", "torque_nm", "radial_force_n"]


def mean_abs_shap(model, X):
    """Mean |SHAP| per feature, averaged over samples and classes."""
    values = shap.TreeExplainer(model).shap_values(X)
    values = np.stack(values, axis=-1) if isinstance(values, list) else values  # -> (n, features, classes)
    return pd.Series(np.abs(values).mean(axis=(0, 2)), index=X.columns)


def shap_report(shap_runs, windows, features, out):
    runs = pd.DataFrame(shap_runs)
    report = pd.DataFrame({"mean_abs_shap": runs.mean(), "std_across_splits": runs.std()})
    report["family"] = [feature_family(f) for f in report.index]
    # Does an important feature mostly encode the operating condition? High |rho| = yes.
    for var in CONDITION_VARIABLES:
        report[f"spearman_{var}"] = [windows[f].corr(windows[var], method="spearman") for f in report.index]
    report = report.sort_values("mean_abs_shap", ascending=False)
    report.index.name = "feature"
    report.to_csv(out / "shap_importance.csv")

    top = report.head(15).iloc[::-1]
    family_colors = dict(zip(["envelope", "time", "band"], MODEL_COLORS.values()))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top["mean_abs_shap"], xerr=top["std_across_splits"], height=0.6,
            color=[family_colors[f] for f in top["family"]], error_kw={"elinewidth": 1, "ecolor": "#52514e"})
    ax.legend(handles=[Patch(color=c, label=f) for f, c in family_colors.items()], frameon=False, fontsize=8,
              title="feature family", title_fontsize=8, loc="lower right")
    ax.set_title("LightGBM mean |SHAP|, bearing-wise splits (mean ± std)", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "shap_importance.png", dpi=150)
    plt.close(fig)

    share = report.groupby("family")["mean_abs_shap"].sum() / report["mean_abs_shap"].sum()
    print("SHAP share by feature family:\n" + share.round(3).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(cfg["paths"]["index"])
    labelled = index.dropna(subset=["label"])
    recordings = labelled[labelled["damage_type"].isin(cfg["dataset"]["damage_types"])].reset_index(drop=True)
    print(f"{len(recordings)} recordings from {recordings['bearing_id'].nunique()} bearings "
          f"({len(index) - len(labelled)} combined-damage recordings excluded, "
          f"damage types: {cfg['dataset']['damage_types']})")

    orders = defect_orders(BearingGeometry(**cfg["geometry"]))
    windows = feature_table(recordings, cfg, orders)
    features = feature_columns(windows)
    y = windows["label"].map(CLASSES.index).to_numpy()

    rows, shap_runs = [], []
    for split_type in cfg["split_types"]:
        for split in make_splits(split_type, recordings, windows, cfg, seed):
            if split.level == "window":
                train = np.zeros(len(windows), dtype=bool)
                train[split.train] = True
            else:
                train = windows["recording_id"].isin(split.train).to_numpy()
            test = ~train
            if split_type == "bearing":  # belt and braces on top of the unit test
                assert not shared_groups(recordings, split, "bearing_id")

            for model_name in cfg["models"]:
                model = fit_model(model_name, windows.loc[train, features], y[train], windows[train],
                                  split_type, cfg, seed)
                assert list(model.classes_) == [0, 1, 2], "training data must contain every class"
                proba = model.predict_proba(windows.loc[test, features])
                rec_y, rec_proba = to_recording_level(windows.loc[test, "recording_id"], y[test], proba)
                common = {"split_type": split_type, "repeat": split.repeat, "model": model_name,
                          "n_train_windows": int(train.sum()), "n_test_windows": int(test.sum()),
                          "best_params": str(getattr(model, "best_params_", ""))}
                rows.append({**common, "level": "window", **classification_metrics(y[test], proba)})
                rows.append({**common, "level": "recording", **classification_metrics(rec_y, rec_proba)})
                if model_name == "lightgbm" and split_type == "bearing":
                    shap_runs.append(mean_abs_shap(model.best_estimator_, windows.loc[test, features]))
            print(f"done: {split_type} split {split.repeat}")

    per_split = pd.DataFrame(rows)
    per_split.to_csv(out / "metrics_per_split.csv", index=False)
    summary = summarise(per_split)
    summary.to_csv(out / "leakage_ladder.csv", index=False)
    for level in ("window", "recording"):
        plot_ladder(summary, out / f"leakage_ladder_{level}.png", level)
    if shap_runs:
        shap_report(shap_runs, windows, features, out)

    print(summary[["split_type", "model", "level", "macro_f1_mean", "macro_f1_std",
                   "macro_auroc_mean", "n_splits"]].to_string(index=False))

    # Project rule: a too-good leakage-free score is a bug until proven otherwise.
    alarm = summary.query("split_type == 'bearing' and model == 'lightgbm'")
    if (alarm["macro_f1_mean"] > cfg["leakage_alarm_f1"]).any():
        print(f"WARNING: bearing-wise macro-F1 above {cfg['leakage_alarm_f1']}. Suspect leakage and "
              "investigate before reporting.")


if __name__ == "__main__":
    main()
