"""
02_feature_engineering.py
──────────────────────────
Reads labeled CSVs → engineers features → produces one merged dataset.

Engineered features (no pressure):
  alt_diff          Δaltitude per timestep
  vel_diff          Δvelocity per timestep  ≈ acceleration
  acc_proxy         second Δ of altitude
  alt_rolling_mean  rolling mean of altitude
  vel_rolling_mean  rolling mean of velocity
  alt_rolling_std   rolling std  of altitude
  vel_rolling_std   rolling std  of velocity
  speed_abs         |velocity|  (symmetric, useful near apogee)
  is_ascending      1 if velocity > 0 else 0

flight_id is added for Leave-One-Rocket-Out CV.
"""

import os
import numpy as np
import pandas as pd
import config
import utils


def engineer_features(df: pd.DataFrame, flight_id: int) -> pd.DataFrame:
    df  = df.copy()
    df.columns = df.columns.str.strip()
    w   = config.ROLLING_WINDOW

    alt = df[config.COL_ALTITUDE]
    vel = df[config.COL_VELOCITY]

    # Diff-based
    df["alt_diff"]  = alt.diff().fillna(0)
    df["vel_diff"]  = vel.diff().fillna(0)
    df["acc_proxy"] = df["alt_diff"].diff().fillna(0)

    # Rolling statistics
    df["alt_rolling_mean"] = alt.rolling(w, min_periods=1).mean()
    df["vel_rolling_mean"] = vel.rolling(w, min_periods=1).mean()
    df["alt_rolling_std"]  = alt.rolling(w, min_periods=1).std().fillna(0)
    df["vel_rolling_std"]  = vel.rolling(w, min_periods=1).std().fillna(0)

    # Derived scalars
    df["speed_abs"]    = vel.abs()
    df["is_ascending"] = (vel > 0).astype(int)

    # Ensure pyro columns exist (default 0 if missing)
    if config.COL_PYRO1 not in df.columns:
        df[config.COL_PYRO1] = 0
    if config.COL_PYRO2 not in df.columns:
        df[config.COL_PYRO2] = 0

    df["flight_id"] = flight_id
    return df


def main():
    utils.ensure_dirs()
    print("\n" + "="*55)
    print("  STEP 2 — Feature Engineering")
    print("="*55)

    labeled_files = sorted([
        f for f in os.listdir(config.LABELED_DIR)
        if f.endswith("_labeled.csv")
    ])
    if not labeled_files:
        raise FileNotFoundError(
            "No labeled CSVs found. Run 01_label_data.py first."
        )

    all_dfs = []
    for fid, fname in enumerate(labeled_files):
        path = os.path.join(config.LABELED_DIR, fname)
        df   = pd.read_csv(path)
        print(f"\n  [{fid}] {fname}  ({len(df)} rows)")
        df_feat = engineer_features(df, flight_id=fid)
        all_dfs.append(df_feat)

    merged = pd.concat(all_dfs, ignore_index=True)

    feature_cols_present = [c for c in config.FEATURE_COLS if c in merged.columns]
    before = len(merged)
    merged.dropna(subset=feature_cols_present, inplace=True)
    print(f"\n  Dropped {before - len(merged)} NaN rows → {len(merged)} total samples")

    # Class distribution
    print("\n  Class distribution:")
    counts = merged[config.LABEL_COL].value_counts()
    max_n  = counts.max()
    for phase in config.PHASE_LABELS:
        n   = counts.get(phase, 0)
        bar = "█" * max(1, int(n / max(max_n // 30, 1)))
        print(f"    {phase:10s}: {n:6d}  {bar}")

    merged.to_csv(config.FEATURES_PATH, index=False)
    print(f"\n✓ Saved → {config.FEATURES_PATH}  shape={merged.shape}")
    print(f"  Features: {feature_cols_present}")


if __name__ == "__main__":
    main()
