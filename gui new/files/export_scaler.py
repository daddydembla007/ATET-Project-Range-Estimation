"""
export_scaler.py
================
Run this ONCE after training to save the StandardScaler.
Place feature_scaler.pkl in the same folder as main.py.

Usage:
    python export_scaler.py
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ── Paste your training data loading here ──────────────────
# This must be the SAME X_train you used during training.


# The scaler must be fit only on training data.

# Example: load from your saved numpy arrays
# X_train = np.load("X_train.npy")   # shape (N, 100, 4)

# ── If you already have the fitted scaler object in memory ──
# after training, just do:
import joblib
joblib.dump(scaler, "feature_scaler.pkl")
# and skip this script.

# ── Refit from scratch (if you don't have saved arrays) ────
# Uncomment and adapt:

# import pandas as pd, os
# from sklearn.preprocessing import StandardScaler
# FEATURES = ['speed_mps','Gradient','Speed Limit[km/h]','Elevation Smoothed[m]']
# CLEAN_FOLDER = "clean_eved"
# all_data = []
# for f in os.listdir(CLEAN_FOLDER):
#     if not f.endswith(".csv"): continue
#     df = pd.read_csv(os.path.join(CLEAN_FOLDER, f), low_memory=False)
#     if 'Speed Limit[km/h]' in df.columns:
#         df['Speed Limit[km/h]'] = pd.to_numeric(
#             df['Speed Limit[km/h]'].astype(str).str.extract(r'(\d+)')[0],
#             errors='coerce').fillna(50.0)
#     cols = [c for c in FEATURES if c in df.columns]
#     all_data.append(df[cols].values)
# all_data = np.vstack(all_data)
# scaler = StandardScaler()
# scaler.fit(all_data)
# joblib.dump(scaler, "feature_scaler.pkl")
# print("Saved feature_scaler.pkl")
# print("Mean:", scaler.mean_)
# print("Scale:", scaler.scale_)

# ── Quick check ─────────────────────────────────────────────
def verify_scaler(path="feature_scaler.pkl"):
    scaler = joblib.load(path)
    print("Scaler loaded successfully")
    print(f"  n_features_in : {scaler.n_features_in_}")
    print(f"  mean          : {scaler.mean_}")
    print(f"  scale         : {scaler.scale_}")
    # Test transform
    dummy = np.array([[10.0, 0.01, 50.0, 200.0]], dtype=np.float32)
    out   = scaler.transform(dummy)
    print(f"  Test input    : {dummy}")
    print(f"  Test output   : {out}")

if __name__ == "__main__":
    verify_scaler()
