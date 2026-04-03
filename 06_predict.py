"""
06_predict.py
──────────────
Given a new flight CSV, predicts the flight phase for every row.
Includes SHAP and LIME explainability for research analysis.

Usage:
    python 06_predict.py --csv your_flight.csv --model xgboost
    python 06_predict.py --csv your_flight.csv --model randomforest --explain_row 100

Model options: xgboost | randomforest | svm | lstm

Outputs:
    - Terminal: phase distribution + sample rows
    - CSV:  outputs/predicted_<flight>_<model>.csv
    - Plot: outputs/plots/prediction_timeline_<flight>.png
    - SHAP: outputs/plots/shap_summary_<flight>.png
            outputs/plots/shap_row<N>_<flight>.png
    - LIME: outputs/plots/lime_row<N>_<flight>.png
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

import config


# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (must exactly match 02_feature_engineering.py)
# ══════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    w = config.ROLLING_WINDOW

    alt = df[config.COL_ALTITUDE]
    vel = df[config.COL_VELOCITY]

    df["alt_diff"]  = alt.diff().fillna(0)
    df["vel_diff"]  = vel.diff().fillna(0)
    df["acc_proxy"] = df["alt_diff"].diff().fillna(0)

    df["alt_rolling_mean"] = alt.rolling(w, min_periods=1).mean()
    df["vel_rolling_mean"] = vel.rolling(w, min_periods=1).mean()
    df["alt_rolling_std"]  = alt.rolling(w, min_periods=1).std().fillna(0)
    df["vel_rolling_std"]  = vel.rolling(w, min_periods=1).std().fillna(0)

    df["speed_abs"]    = vel.abs()
    df["is_ascending"] = (vel > 0).astype(int)

    if config.COL_PYRO1 not in df.columns:
        df[config.COL_PYRO1] = 0
    if config.COL_PYRO2 not in df.columns:
        df[config.COL_PYRO2] = 0

    return df


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

def find_model_path(model_name: str) -> str:
    prefix   = model_name.lower().replace(" ", "_")
    all_pkls = [
        f for f in os.listdir(config.MODELS_DIR)
        if f.startswith(prefix) and f.endswith(".pkl")
    ]
    if not all_pkls:
        raise FileNotFoundError(
            f"No saved model found for '{model_name}' in {config.MODELS_DIR}/\n"
            f"Run 03_train_evaluate.py first."
        )
    preferred = [f for f in all_pkls if "fold_rocket_1" in f]
    chosen    = preferred[0] if preferred else sorted(all_pkls)[0]
    return os.path.join(config.MODELS_DIR, chosen)


def load_model(model_name: str):
    path  = find_model_path(model_name)
    saved = joblib.load(path)
    print(f"  Loaded model : {os.path.basename(path)}")
    return saved["model"], saved["scaler"], saved["le"], saved["feature_cols"]


# ══════════════════════════════════════════════════════════════════════
# LSTM PREDICTION
# ══════════════════════════════════════════════════════════════════════

def predict_lstm(X_scaled: np.ndarray) -> list:
    import tensorflow as tf
    seq_len = config.LSTM_SEQUENCE_LEN

    keras_files = [
        f for f in os.listdir(config.MODELS_DIR)
        if f.startswith("lstm_") and f.endswith(".keras")
    ]
    if not keras_files:
        raise FileNotFoundError("No LSTM .keras model found. Run 04_lstm_model.py first.")

    preferred  = [f for f in keras_files if "rocket_1" in f]
    chosen     = preferred[0] if preferred else sorted(keras_files)[0]
    model      = tf.keras.models.load_model(os.path.join(config.MODELS_DIR, chosen))
    print(f"  Loaded model : {chosen}")

    sequences   = np.array([X_scaled[i-seq_len:i] for i in range(seq_len, len(X_scaled))])
    probs       = model.predict(sequences, verbose=0)
    pred_labels = [config.PHASE_LABELS[i] for i in np.argmax(probs, axis=1)]
    return [pred_labels[0]] * seq_len + pred_labels


# ══════════════════════════════════════════════════════════════════════
# PREDICTION TIMELINE PLOT
# ══════════════════════════════════════════════════════════════════════

def plot_prediction_timeline(df: pd.DataFrame, flight_name: str, model_name: str):
    phase_colors = {
        "Boost":   "#e74c3c", "Coast":   "#f39c12",
        "Apogee":  "#9b59b6", "Descent": "#2980b9", "Landed":  "#27ae60",
    }
    fig, ax = plt.subplots(figsize=(13, 4))
    for phase, color in phase_colors.items():
        mask = df["predicted_phase"] == phase
        ax.scatter(df.loc[mask, config.COL_TIME], df.loc[mask, config.COL_ALTITUDE],
                   c=color, label=phase, s=6, alpha=0.85)
    ax.set_xlabel(f"Time ({config.COL_TIME})")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Predicted Phases — {flight_name}  [{model_name.upper()}]",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"prediction_timeline_{flight_name}_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Timeline plot → {path}")


# ══════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════

def run_shap(model, X_scaled: np.ndarray, feature_cols: list,
             flight_name: str, explain_row: int):
    """
    SHAP asks: how much did each feature push this prediction away from average?
    Positive SHAP = pushed TOWARD the predicted class.
    Negative SHAP = pushed AWAY from the predicted class.

    Two plots:
      1. Summary (global)  — which features matter most across the whole flight
      2. Waterfall (local) — which features drove the prediction for ONE specific row
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Run: pip install shap")
        return

    print("\n  Running SHAP ...")
    model_type = type(model).__name__

    if "XGB" in model_type or "Forest" in model_type or "Tree" in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer  = shap.KernelExplainer(model.predict_proba, background)

    # Sample for speed
    sample_size = min(500, len(X_scaled))
    idx_sample  = np.random.choice(len(X_scaled), sample_size, replace=False)
    shap_values = explainer(X_scaled[idx_sample])

    vals = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    mean_abs = np.abs(vals).mean(axis=0)
    if mean_abs.ndim == 2:       # multiclass: (n_features, n_classes)
        mean_abs = mean_abs.mean(axis=1)

    # ── Global summary bar chart ─────────────────────────────────────
    order      = np.argsort(mean_abs)
    fig, ax    = plt.subplots(figsize=(9, 6))
    colors     = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feature_cols)))
    ax.barh([feature_cols[i] for i in order], mean_abs[order], color=colors)
    ax.set_xlabel("Mean |SHAP value|  —  average impact on phase prediction")
    ax.set_title(f"SHAP Global Feature Importance\n{flight_name}  [{model_type}]",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"shap_summary_{flight_name}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  SHAP summary  → {path}")

    # ── Single-row waterfall ─────────────────────────────────────────
    row_idx   = min(explain_row, len(X_scaled) - 1)
    row_shap  = explainer(X_scaled[row_idx:row_idx+1])
    row_vals  = row_shap.values[0] if hasattr(row_shap, "values") else np.array(row_shap)[0]

    pred_proba      = model.predict_proba(X_scaled[row_idx:row_idx+1])[0]
    pred_class_idx  = int(np.argmax(pred_proba))
    pred_class_name = config.PHASE_LABELS[pred_class_idx]

    if row_vals.ndim == 2:
        row_vals = row_vals[:, pred_class_idx]

    top_n    = 12
    order_r  = np.argsort(np.abs(row_vals))[::-1][:top_n]
    names_r  = [feature_cols[i] for i in order_r]
    vals_r   = row_vals[order_r]
    colors_r = ["#e74c3c" if v > 0 else "#2980b9" for v in vals_r]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names_r[::-1], vals_r[::-1], color=colors_r[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (red = supports predicted class | blue = opposes)")
    ax.set_title(f"SHAP Row {row_idx} Explanation\n"
                 f"Predicted: {pred_class_name}  (confidence: {pred_proba[pred_class_idx]*100:.1f}%)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"shap_row{row_idx}_{flight_name}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  SHAP waterfall→ {path}")


# ══════════════════════════════════════════════════════════════════════
# LIME EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════

def run_lime(model, X_scaled: np.ndarray, feature_cols: list,
             flight_name: str, explain_row: int):
    """
    LIME explains ONE row at a time using a local linear approximation.

    Steps internally:
      1. Takes your row
      2. Creates 1000 slightly perturbed copies of it
      3. Gets predictions for all copies
      4. Fits a simple linear model on those copies
      5. The linear weights = which features mattered for THIS prediction

    Red bars = feature condition that supports the predicted phase
    Blue bars = feature condition that opposes the predicted phase
    The inset shows confidence % across all 5 phases.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("  LIME not installed. Run: pip install lime")
        return

    print("\n  Running LIME ...")
    row_idx         = min(explain_row, len(X_scaled) - 1)
    pred_proba      = model.predict_proba(X_scaled[row_idx:row_idx+1])[0]
    pred_class_idx  = int(np.argmax(pred_proba))
    pred_class_name = config.PHASE_LABELS[pred_class_idx]

    explainer = LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=feature_cols,
        class_names=config.PHASE_LABELS,
        mode="classification",
        discretize_continuous=True,
        random_state=config.RANDOM_STATE,
    )

    explanation = explainer.explain_instance(
        data_row=X_scaled[row_idx],
        predict_fn=model.predict_proba,
        num_features=10,
        top_labels=1,
        num_samples=1000,
    )

    lime_list    = explanation.as_list(label=pred_class_idx)
    feat_labels  = [item[0] for item in lime_list]
    feat_weights = [item[1] for item in lime_list]
    colors       = ["#e74c3c" if w > 0 else "#2980b9" for w in feat_weights]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(feat_labels[::-1], feat_weights[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight  (red = supports predicted class | blue = opposes)")
    ax.set_title(
        f"LIME Explanation — Row {row_idx}\n"
        f"Predicted: {pred_class_name}  (confidence: {pred_proba[pred_class_idx]*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Inset: confidence across all 5 phases
    ax2 = ax.inset_axes([0.63, 0.02, 0.35, 0.38])
    bar_colors = ["#e74c3c","#f39c12","#9b59b6","#2980b9","#27ae60"]
    ax2.barh(config.PHASE_LABELS, pred_proba, color=bar_colors)
    ax2.set_xlim(0, 1)
    ax2.set_title("Class confidence", fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"lime_row{row_idx}_{flight_name}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  LIME plot     → {path}")

    # Terminal explanation
    print(f"\n  LIME text explanation for row {row_idx}:")
    print(f"  Predicted: {pred_class_name}  ({pred_proba[pred_class_idx]*100:.1f}% confidence)")
    print(f"  {'Condition':<42} {'Weight':>8}  Direction")
    print(f"  {'-'*60}")
    for label, weight in lime_list:
        direction = "FOR  " if weight > 0 else "AGAINST"
        print(f"  {label:<42} {weight:>+8.4f}  {direction}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Predict flight phases + SHAP/LIME explainability."
    )
    parser.add_argument("--csv",         required=True,  help="Path to flight CSV")
    parser.add_argument("--model",       default="xgboost",
                        choices=["xgboost","randomforest","svm","lstm"])
    parser.add_argument("--explain_row", type=int, default=50,
                        help="Row index to explain with SHAP & LIME (default: 50)")
    parser.add_argument("--no_shap",     action="store_true", help="Skip SHAP")
    parser.add_argument("--no_lime",     action="store_true", help="Skip LIME")
    args = parser.parse_args()

    os.makedirs(config.PLOTS_DIR,  exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Load CSV ─────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df          = pd.read_csv(args.csv)
    df.columns  = df.columns.str.strip()
    flight_name = os.path.splitext(os.path.basename(args.csv))[0]

    print(f"\n{'='*55}")
    print(f"  Flight : {flight_name}")
    print(f"  Model  : {args.model.upper()}")
    print(f"  Rows   : {len(df)}")
    print(f"{'='*55}")

    # ── Feature engineering ──────────────────────────────────────────
    df_feat      = engineer_features(df)
    feature_cols = [c for c in config.FEATURE_COLS if c in df_feat.columns]
    X            = df_feat[feature_cols].fillna(0).values

    # ── Predict ──────────────────────────────────────────────────────
    if args.model == "lstm":
        from sklearn.preprocessing import StandardScaler
        X_scaled    = StandardScaler().fit_transform(X)
        predictions = predict_lstm(X_scaled)
        args.no_shap = True
        args.no_lime = True
        print("  Note: SHAP/LIME not supported for LSTM. Use xgboost or randomforest.")
        model = None
    else:
        model, scaler, le, saved_cols = load_model(args.model)
        if saved_cols:
            feature_cols = [c for c in saved_cols if c in df_feat.columns]
            X = df_feat[feature_cols].fillna(0).values
        X_scaled    = scaler.transform(X)
        predictions = le.inverse_transform(model.predict(X_scaled)).tolist()

    # ── Attach predictions ────────────────────────────────────────────
    df["predicted_phase"] = predictions

    # ── Terminal summary ──────────────────────────────────────────────
    print(f"\n  Phase distribution:")
    counts = df["predicted_phase"].value_counts()
    for phase in config.PHASE_LABELS:
        n   = counts.get(phase, 0)
        pct = 100 * n / len(df)
        print(f"    {phase:10s}: {n:5d} rows  ({pct:5.1f}%)  {'█' * int(pct/2)}")

    print(f"\n  Sample rows (first 10):")
    print(f"  {'Row':<6} {'Time':<12} {'Altitude':>10} {'Velocity':>10}  Phase")
    print(f"  {'-'*55}")
    for i, row in df.head(10).iterrows():
        print(f"  {i:<6} {row[config.COL_TIME]:<12} "
              f"{row[config.COL_ALTITUDE]:>10.1f} "
              f"{row[config.COL_VELOCITY]:>10.1f}  "
              f"{row['predicted_phase']}")

    # ── Save CSV ──────────────────────────────────────────────────────
    out_csv = os.path.join(config.OUTPUT_DIR,
                           f"predicted_{flight_name}_{args.model}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Output CSV    → {out_csv}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_prediction_timeline(df, flight_name, args.model)

    if not args.no_shap and model is not None:
        run_shap(model, X_scaled, feature_cols, flight_name, args.explain_row)

    if not args.no_lime and model is not None:
        run_lime(model, X_scaled, feature_cols, flight_name, args.explain_row)

    print(f"\n{'='*55}")
    print(f"  All outputs saved to outputs/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()