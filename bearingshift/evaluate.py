"""Metrics, recording-level aggregation, mean +/- std summaries and the ladder plot."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from bearingshift import CLASSES
from bearingshift.splits import SPLIT_TYPES

METRICS = ["macro_auroc", "macro_f1"] + [f"recall_{c}" for c in CLASSES]
LADDER_LABELS = {"random_window": "a. random\nwindow", "recording": "b. recording",
                 "condition": "c. condition", "bearing": "d. bearing\n(leakage-free)"}
# Fixed categorical order, one hue per model (validated reference palette, light mode).
MODEL_COLORS = {"majority": "#2a78d6", "physics_rule": "#eb6834", "logistic_time": "#1baf7a", "lightgbm": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def classification_metrics(y_true, proba):
    """Macro one-vs-rest AUROC, macro-F1 and per-class recall.

    Macro = every class counts equally, whatever its size. AUROC for a class is skipped
    (NaN) if that class is absent from y_true, because it is undefined then.
    """
    y_true = np.asarray(y_true)
    pred = proba.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    aurocs = [roc_auc_score(y_true == k, proba[:, k]) for k in labels if 0 < (y_true == k).sum() < len(y_true)]
    recalls = recall_score(y_true, pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else np.nan,
        "macro_f1": f1_score(y_true, pred, labels=labels, average="macro", zero_division=0),
        **{f"recall_{c}": r for c, r in zip(CLASSES, recalls)},
    }


def to_recording_level(recording_ids, y_true, proba):
    """Average window probabilities within each recording (one decision per recording)."""
    df = pd.DataFrame(proba, columns=CLASSES)
    df["recording_id"], df["y"] = np.asarray(recording_ids), np.asarray(y_true)
    grouped = df.groupby("recording_id")
    return grouped["y"].first().to_numpy(), grouped[CLASSES].mean().to_numpy()


def summarise(per_split):
    """mean, std and number of splits for every metric, per split type x model x level."""
    agg = per_split.groupby(["split_type", "model", "level"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_splits"] = per_split.groupby(["split_type", "model", "level"]).size()
    agg = agg.reset_index()
    agg["split_type"] = pd.Categorical(agg["split_type"], SPLIT_TYPES, ordered=True)
    return agg.sort_values(["level", "model", "split_type"]).reset_index(drop=True)


def plot_ladder(summary, path, level="window"):
    """Two small multiples (macro-F1, macro-AUROC): each model's mean +/- std per ladder rung."""
    data = summary[summary["level"] == level]
    splits = [s for s in SPLIT_TYPES if s in set(data["split_type"])]
    x = np.arange(len(splits))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric, title in zip(axes, ["macro_f1", "macro_auroc"], ["Macro-F1", "Macro-AUROC"]):
        for model, color in MODEL_COLORS.items():
            rows = data[data["model"] == model].set_index("split_type").reindex(splits)
            if rows[f"{metric}_mean"].isna().all():
                continue
            ax.errorbar(x, rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"], color=color, lw=2,
                        marker="o", ms=8, mec="white", mew=2, capsize=0, elinewidth=2, label=model)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xticks(x, [LADDER_LABELS[s] for s in splits], fontsize=8, color=MUTED)
        ax.set_xlim(-0.3, len(splits) - 0.3)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=8, labelcolor=MUTED)
    fig.suptitle(f"Leakage ladder ({level} level): mean ± std across splits", x=0.06, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
