# utils.py — Shared helpers for the pipeline
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
import config


def ensure_dirs():
    for d in [config.OUTPUT_DIR, config.LABELED_DIR,
              config.MODELS_DIR, config.PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def load_flight_csvs(data_dir=config.DATA_DIR):
    """Load all CSVs from data_dir. Returns {stem: DataFrame}."""
    flights = {}
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}/'")
    for f in csv_files:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        flights[os.path.splitext(f)[0]] = df
        print(f"  Loaded: {f}  →  {len(df)} rows, {df.shape[1]} cols")
    return flights


def save_model(obj, name):
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    joblib.dump(obj, path)
    print(f"  Saved model → {path}")


def load_model(name):
    return joblib.load(os.path.join(config.MODELS_DIR, f"{name}.pkl"))


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name, labels=config.PHASE_LABELS):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Phase")
    ax.set_ylabel("True Phase")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def plot_roc_curves(y_true, y_score, model_name, labels=config.PHASE_LABELS):
    """One-vs-Rest ROC curves. y_score: (n_samples, n_classes) probabilities."""
    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for i, label in enumerate(labels):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curves (OvR)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f"roc_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved ROC curves → {path}")


def print_classification_report(y_true, y_pred, model_name):
    print(f"\n{'='*55}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                labels=config.PHASE_LABELS,
                                zero_division=0))


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(n), importances[indices][::-1], color="steelblue")
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Top {n} Feature Importances",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f"feat_imp_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved feature importance → {path}")


def plot_phase_timeline(df, flight_name):
    """Altitude coloured by phase label for visual verification."""
    phase_colors = {
        "Boost":   "#e74c3c",
        "Coast":   "#f39c12",
        "Apogee":  "#9b59b6",
        "Descent": "#2980b9",
        "Landed":  "#27ae60",
    }
    fig, ax = plt.subplots(figsize=(12, 4))
    for phase, color in phase_colors.items():
        mask = df[config.LABEL_COL] == phase
        ax.scatter(df.loc[mask, config.COL_TIME],
                   df.loc[mask, config.COL_ALTITUDE],
                   c=color, label=phase, s=5, alpha=0.8)
    ax.set_xlabel(f"Time ({config.COL_TIME})")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Flight: {flight_name} — Phase Labels",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"timeline_{flight_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved timeline → {path}")
