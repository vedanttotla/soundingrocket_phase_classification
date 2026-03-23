# config.py — Central configuration for the pipeline

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR      = "data"
OUTPUT_DIR    = "outputs"
LABELED_DIR   = os.path.join(OUTPUT_DIR, "labeled")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features_dataset.csv")
MODELS_DIR    = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR     = os.path.join(OUTPUT_DIR, "plots")

# ─────────────────────────────────────────────
# COLUMN NAMES  — exact headers from your CSVs
# ─────────────────────────────────────────────
COL_LINK      = "link"               # Telemetry link ID (1 or 2)
COL_TIME      = "ts[deciseconds]"    # Timestamp in deciseconds
COL_RAW_STATE = "state"              # Numeric state from flight computer
COL_ERRORS    = "errors"
COL_LAT       = "lat[deg/10000]"
COL_LON       = "lon[deg/10000]"
COL_ALTITUDE  = "altitude[m]"        # Altitude in metres
COL_VELOCITY  = "velocity[m/s]"      # Vertical velocity m/s
COL_BATTERY   = "battery[decivolts]"
COL_PYRO1     = "pyro1"              # 0/1 ejection charge fired
COL_PYRO2     = "pyro2"              # 0/1 ejection charge fired
# NOTE: No pressure sensor — not used anywhere in this pipeline

# ─────────────────────────────────────────────
# NUMERIC STATE → PHASE MAPPING
# Flight computer codes observed in data:
#   3 = Boost  |  4 = Coast (will be split into Coast/Apogee by labeler)
#   5 = Descent  |  6 = Landed
# ─────────────────────────────────────────────
RAW_STATE_MAP = {
    3: "Boost",
    4: "Coast",
    5: "Descent",
    6: "Landed",
}

# Feature columns used for ML (time, raw state, errors, link excluded)
FEATURE_COLS = [
    COL_ALTITUDE,       # altitude[m]
    COL_VELOCITY,       # velocity[m/s]
    COL_PYRO1,          # pyro1
    COL_PYRO2,          # pyro2
    COL_LAT,            # lat[deg/10000]
    COL_LON,            # lon[deg/10000]
    COL_BATTERY,        # battery[decivolts]
    # Engineered features (added by 02_feature_engineering.py):
    "alt_diff",
    "vel_diff",
    "acc_proxy",
    "alt_rolling_mean",
    "vel_rolling_mean",
    "alt_rolling_std",
    "vel_rolling_std",
    "speed_abs",
    "is_ascending",
]

# ─────────────────────────────────────────────
# LABELING THRESHOLDS
# Tuned to actual data: vel -39→+229 m/s, alt 0→3466 m
# ─────────────────────────────────────────────
APOGEE_VEL_THRESHOLD = 10.0
LANDED_ALT_THRESHOLD = 20.0
LANDED_VEL_THRESHOLD =  5.0
BOOST_VEL_MIN        = 30.0
APOGEE_WINDOW_FRAC   =  0.04

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
ROLLING_WINDOW = 5

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
SPLIT_MODE   = "leave_one_out"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":           6,
    "learning_rate":    0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":    "mlogloss",
    "random_state":  RANDOM_STATE,
    "n_jobs":             -1,
}

RF_PARAMS = {
    "n_estimators":      300,
    "max_depth":        None,
    "min_samples_split":   5,
    "random_state":  RANDOM_STATE,
    "n_jobs":             -1,
}

SVM_PARAMS = {
    "kernel":  "rbf",
    "C":        10.0,
    "gamma":  "scale",
    "decision_function_shape": "ovr",
}

# ─────────────────────────────────────────────
# LSTM HYPERPARAMETERS
# ─────────────────────────────────────────────
LSTM_SEQUENCE_LEN = 20
LSTM_UNITS        = [64, 32]
LSTM_DROPOUT      = 0.3
LSTM_BATCH_SIZE   = 32
LSTM_EPOCHS       = 50
LSTM_PATIENCE     = 10

# ─────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────
PHASE_LABELS = ["Boost", "Coast", "Apogee", "Descent", "Landed"]
LABEL_COL    = "phase"
