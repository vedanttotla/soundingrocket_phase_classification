"""
06_predict.py
──────────────
Given a new flight CSV, predicts the phase for every row.

Usage:
    python 06_predict.py --csv your_flight.csv --model xgboost

Model options: xgboost | randomforest | svm | lstm

Output:
    - Prints phase for each row in the terminal
    - Saves a new CSV with a 'predicted_phase' column
    - Saves a timeline plot (altitude coloured by predicted phase)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import config


# ──────────────────────────────────────────────
# Feature engineering (must match 02_feature_engineering.py)
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Find best saved model for a given model type
# (picks the fold with the best pkl available)
# ──────────────────────────────────────────────

def find_model_path(model_name: str) -> str:
    """
    Looks for saved model files in outputs/models/.
    Prefers fold_rocket_1 onwards (fold 0 had poor labels).
    Falls back to any available fold.
    """
    prefix = model_name.lower().replace(" ", "_")
    all_pkls = [
        f for f in os.listdir(config.MODELS_DIR)
        if f.startswith(prefix) and f.endswith(".pkl")
    ]
    if not all_pkls:
        raise FileNotFoundError(
            f"No saved model found for '{model_name}' in {config.MODELS_DIR}/\n"
            f"Run 03_train_evaluate.py first."
        )
    # Prefer fold_rocket_1 as it had the best scores
    preferred = [f for f in all_pkls if "fold_rocket_1" in f]
    chosen = preferred[0] if preferred else sorted(all_pkls)[0]
    return os.path.join(config.MODELS_DIR, chosen)


# ──────────────────────────────────────────────
# LSTM prediction
# ──────────────────────────────────────────────

def predict_lstm(X_scaled, n_samples_original):
    import tensorflow as tf
    seq_len = config.LSTM_SEQUENCE_LEN

    # Find best keras model
    keras_files = [
        f for f in os.listdir(config.MODELS_DIR)
        if f.startswith("lstm_") and f.endswith(".keras")
    ]
    if not keras_files:
        raise FileNotFoundError("No LSTM .keras model found. Run 04_lstm_model.py first.")

    preferred = [f for f in keras_files if "rocket_1" in f]
    chosen = preferred[0] if preferred else sorted(keras_files)[0]
    model_path = os.path.join(config.MODELS_DIR, chosen)
    print(f"  Using LSTM model: {chosen}")

    model = tf.keras.models.load_model(model_path)

    # Build sequences
    sequences = []
    for i in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[i - seq_len:i])
    sequences = np.array(sequences)

    probs      = model.predict(sequences, verbose=0)
    pred_enc   = np.argmax(probs, axis=1)
    pred_labels = [config.PHASE_LABELS[i] for i in pred_enc]

    # Pad first seq_len rows with the first predicted label
    padding = [pred_labels[0]] * seq_len
    return padding + pred_labels


# ──────────────────────────────────────────────
# Plot predicted phases
# ──────────────────────────────────────────────

def plot_prediction(df, flight_name, model_name):
    phase_colors = {
        "Boost":   "#e74c3c",
        "Coast":   "#f39c12",
        "Apogee":  "#9b59b6",
        "Descent": "#2980b9",
        "Landed":  "#27ae60",
    }
    fig, ax = plt.subplots(figsize=(13, 4))
    for phase, color in phase_colors.items():
        mask = df["predicted_phase"] == phase
        ax.scatter(df.loc[mask, config.COL_TIME],
                   df.loc[mask, config.COL_ALTITUDE],
                   c=color, label=phase, s=6, alpha=0.85)

    ax.set_xlabel(f"Time ({config.COL_TIME})")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Predicted Phases — {flight_name}  [{model_name}]",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(
        config.PLOTS_DIR,
        f"prediction_{flight_name}_{model_name.lower()}.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved → {plot_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict flight phases for a new rocket CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to your flight CSV file"
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "randomforest", "svm", "lstm"],
        help="Which model to use for prediction (default: xgboost)"
    )
    args = parser.parse_args()

    # ── Load CSV ────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    flight_name = os.path.splitext(os.path.basename(args.csv))[0]
    print(f"\n{'='*55}")
    print(f"  Predicting phases for: {flight_name}")
    print(f"  Model: {args.model.upper()}")
    print(f"  Rows: {len(df)}")
    print(f"{'='*55}")

    # ── Feature engineering ─────────────────────────────────────────
    df_feat = engineer_features(df)
    feature_cols = [c for c in config.FEATURE_COLS if c in df_feat.columns]
    X = df_feat[feature_cols].fillna(0).values

    # ── Predict ─────────────────────────────────────────────────────
    if args.model == "lstm":
        # LSTM handles its own scaling internally via the saved scaler
        model_path = find_model_path("lstm")
        saved      = joblib.load(model_path) if model_path.endswith(".pkl") else None
        scaler     = saved["scaler"] if saved else None

        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)

        predictions = predict_lstm(X_scaled, len(df))

    else:
        model_path = find_model_path(args.model)
        print(f"  Using model: {os.path.basename(model_path)}")
        saved      = joblib.load(model_path)
        model      = saved["model"]
        scaler     = saved["scaler"]
        le         = saved["le"]

        X_scaled    = scaler.transform(X)
        pred_enc    = model.predict(X_scaled)
        predictions = le.inverse_transform(pred_enc).tolist()

    # ── Attach predictions to original dataframe ────────────────────
    df["predicted_phase"] = predictions

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n  Phase distribution in this flight:")
    counts = df["predicted_phase"].value_counts()
    for phase in config.PHASE_LABELS:
        n   = counts.get(phase, 0)
        pct = 100 * n / len(df)
        bar = "█" * int(pct / 2)
        print(f"    {phase:10s}: {n:5d} rows  ({pct:5.1f}%)  {bar}")

    # ── Print first few rows ─────────────────────────────────────────
    print(f"\n  Sample output (first 10 rows):")
    print(f"  {'Row':<6} {'Time':<12} {'Altitude':>10} {'Velocity':>10} {'Phase'}")
    print(f"  {'-'*55}")
    for i, row in df.head(10).iterrows():
        print(f"  {i:<6} {row[config.COL_TIME]:<12} "
              f"{row[config.COL_ALTITUDE]:>10.1f} "
              f"{row[config.COL_VELOCITY]:>10.1f} "
              f"  {row['predicted_phase']}")

    # ── Save output CSV ──────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(
        config.OUTPUT_DIR,
        f"predicted_{flight_name}_{args.model}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\n  Full CSV saved → {out_csv}")

    # ── Plot ─────────────────────────────────────────────────────────
    plot_prediction(df, flight_name, args.model)

    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()