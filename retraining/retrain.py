# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:24:52 2025

@author: kehin
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.clustering import fcm_membership_for_points
from models.dea import compute_dea_scores
from models.tabpfn_model import build_and_fit_tabpfn

# Module-level shared config/state initialized by init_runtime()
data = None
mdfcm = None
scaler_cluster = None
data_lock = None
mdfcm_lock = None
output_dir = None
models_dir = None
features = None
cluster_cols = None
seed = 42
device = "cpu"

def init_runtime(shared):
    global data, mdfcm, scaler_cluster, data_lock, mdfcm_lock
    global output_dir, models_dir, features, cluster_cols, seed, device

    data = shared["data"]
    mdfcm = shared["mdfcm"]
    scaler_cluster = shared["scaler_cluster"]
    data_lock = shared["data_lock"]
    mdfcm_lock = shared["mdfcm_lock"]
    output_dir = shared["output_dir"]
    models_dir = shared["models_dir"]
    features = shared["features"]
    cluster_cols = shared["cluster_cols"]
    seed = shared.get("seed", seed)
    device = shared.get("device", device)

def retrain_models():
    """
    Retrain per-cluster TabPFN models after recomputing clusters and DEA labels.
    ADAPTIVE SCALER: refits `scaler_cluster` on ALL current data each retrain.
    Thread-safe: protects shared `data` and `mdfcm` with locks.
    Rebuilds and saves a fresh trained_models_index.pkl and the updated scaler.
    """
    global data, mdfcm, scaler_cluster
    print("[Retraining] Starting model retraining with updated data")
    try:
        with data_lock:
            X_cluster = data[cluster_cols].values.copy()

        with mdfcm_lock:
            scaler_cluster = StandardScaler().fit(X_cluster)
            X_scaled = scaler_cluster.transform(X_cluster)

            mdfcm.fit(X_scaled)
            U_all = fcm_membership_for_points(mdfcm.X_all_, mdfcm.V, m=mdfcm.m)
            labels_all = np.argmax(U_all, axis=1) + 1

        with data_lock:
            data['Cluster'] = labels_all
            snapshot = data[['Cluster'] + features].copy()

        trained_models = {}

        for cluster_id in range(1, mdfcm.c + 1):
            cluster_df = snapshot[snapshot['Cluster'] == cluster_id].copy()
            n_cluster = len(cluster_df)
            if n_cluster < 20:
                print(f"[Retraining] Cluster {cluster_id} too small ({n_cluster}). Skipping.")
                continue

            eff_scores = compute_dea_scores(cluster_df)
            cluster_df['Efficiency Score (E)'] = eff_scores
            cluster_df = cluster_df.dropna(subset=['Efficiency Score (E)']).reset_index(drop=True)
            if len(cluster_df) < 20:
                print(f"[Retraining] Cluster {cluster_id} has <20 valid rows post-DEA. Skipping.")
                continue

            X = cluster_df[features].values
            y = cluster_df['Efficiency Score (E)'].values.reshape(-1, 1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.10, random_state=seed
            )

            tabpfn_model = build_and_fit_tabpfn(X_train, y_train, device=device)

            model_path = os.path.join(models_dir, f"tabpfn_cluster_{cluster_id}.joblib")
            joblib.dump(tabpfn_model, model_path)

            trained_models[cluster_id] = {
                "model_path": model_path,
                "n_train": int(len(X_train)),
            }

        joblib.dump(scaler_cluster, os.path.join(output_dir, "scaler_cluster.joblib"))
        joblib.dump(mdfcm,          os.path.join(output_dir, "mdfcm_LATEST.pkl"))
        joblib.dump(trained_models, os.path.join(output_dir, "trained_models_index.pkl"))

        print("\n[Retraining] Completed; scaler, clustering, and index refreshed.\n")

    except Exception as e:
        print(f"[Retraining] Error during retraining: {str(e)}")