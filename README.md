# Sounding Rocket Flight Phase Classification

Supervised ML pipeline for classifying sounding rocket flight phases using flight computer sensor data from 8 flights.

## Target Classes
| Phase | Description |
|-------|-------------|
| `Boost` | Motor burning — rapid altitude and velocity gain |
| `Coast` | Motor burnout — still ascending under inertia |
| `Apogee` | Peak altitude — near-zero vertical velocity |
| `Descent` | Falling under parachute |
| `Landed` | On ground — near-zero velocity and altitude |

## Models
- **XGBoost** — Gradient boosted trees
- **Random Forest** — Ensemble tree model
- **SVM** — Support Vector Machine (RBF kernel)
- **LSTM** — Bidirectional LSTM (sequence-aware)

## Input CSV Format (exact columns required)
```
link, ts[deciseconds], state, errors, lat[deg/10000], lon[deg/10000],
altitude[m], velocity[m/s], battery[decivolts], pyro1, pyro2
```
> **No pressure column** — pipeline is fully configured without it.

The `state` column contains raw flight computer codes:
- `3` = Boost, `4` = Coast/Apogee region, `5` = Descent, `6` = Landed

## Project Structure
```
sounding_rocket_classifier/
├── data/                        ← Place your 8 CSV files here
├── outputs/
│   ├── labeled/                 ← Auto-labeled CSVs
│   ├── models/                  ← Saved models (.pkl / .keras)
│   └── plots/                   ← All evaluation plots
├── config.py                    ← All parameters (columns, thresholds, hyperparams)
├── utils.py                     ← Shared helpers
├── 01_label_data.py             ← Physics-based auto-labeling
├── 02_feature_engineering.py    ← Feature extraction & dataset merge
├── 03_train_evaluate.py         ← XGBoost / RF / SVM training
├── 04_lstm_model.py             ← LSTM training
└── 05_compare_models.py         ← Final comparison chart for paper
```

## Quick Start
```bash
pip install -r requirements.txt

# 1. Copy your 8 CSVs into data/
# 2. Run pipeline in order:
python 01_label_data.py          # Auto-label + verify timeline plots
python 02_feature_engineering.py # Feature engineering + merge
python 03_train_evaluate.py      # Train XGBoost, RF, SVM
python 04_lstm_model.py          # Train LSTM
python 05_compare_models.py      # Paper-ready comparison chart
```

## Evaluation Protocol
**Leave-One-Rocket-Out Cross-Validation** — trains on 7 rockets, tests on 1, rotates through all 8. This tests genuine cross-vehicle generalization with no data leakage.

Outputs per model:
- Confusion matrix
- Per-class precision / recall / F1
- ROC/AUC curves (One-vs-Rest)
- Feature importance (XGBoost + RF)
- Learning curves (LSTM)

## Requirements
```
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn joblib
```
