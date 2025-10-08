# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:23:21 2025

@author: kehin
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import threading
import queue
from datetime import datetime

from models.clustering import fcm_membership_for_points
from models.dea import compute_dea_scores

# These globals will be provided by main.py during initialization
data_queue = None
data_lock = None
mdfcm_lock = None
stop_event = None

IDLE_SHUTDOWN_MIN = 10
QUEUE_TIMEOUT_SEC = 300
max_empty_hits = 2  # will be recalculated in init() if needed
empty_hits = 0

# Runtime shared objects
data = None
scaler_cluster = None
mdfcm = None
output_dir = None
features = None
cluster_cols = None

def init_runtime(shared):
    """
    Initialize module-level references to shared objects and config.
    `shared` is a dict containing keys:
    data, scaler_cluster, mdfcm, output_dir, features, cluster_cols,
    data_queue, data_lock, mdfcm_lock, stop_event, IDLE_SHUTDOWN_MIN, QUEUE_TIMEOUT_SEC
    """
    global data, scaler_cluster, mdfcm, output_dir, features, cluster_cols
    global data_queue, data_lock, mdfcm_lock, stop_event
    global IDLE_SHUTDOWN_MIN, QUEUE_TIMEOUT_SEC, max_empty_hits, empty_hits

    data = shared["data"]
    scaler_cluster = shared["scaler_cluster"]
    mdfcm = shared["mdfcm"]
    output_dir = shared["output_dir"]
    features = shared["features"]
    cluster_cols = shared["cluster_cols"]

    data_queue = shared["data_queue"]
    data_lock = shared["data_lock"]
    mdfcm_lock = shared["mdfcm_lock"]
    stop_event = shared["stop_event"]

    IDLE_SHUTDOWN_MIN = shared.get("IDLE_SHUTDOWN_MIN", IDLE_SHUTDOWN_MIN)
    QUEUE_TIMEOUT_SEC = shared.get("QUEUE_TIMEOUT_SEC", QUEUE_TIMEOUT_SEC)
    max_empty_hits = max(1, int(np.ceil((IDLE_SHUTDOWN_MIN*60) / QUEUE_TIMEOUT_SEC)))
    empty_hits = 0

def real_time_prediction_worker():
    print("\n[Real-time] Prediction worker started\n")
    global empty_hits
    while not stop_event.is_set():
        try:
            new_data_path = data_queue.get(timeout=QUEUE_TIMEOUT_SEC)
            empty_hits = 0
        except queue.Empty:
            empty_hits += 1
            print(f"[Real-time] Idle: no new data in {QUEUE_TIMEOUT_SEC/60:.0f} minutes "
                  f"({empty_hits}/{max_empty_hits})")
            if empty_hits >= max_empty_hits:
                print(f"[Real-time] No new data for {IDLE_SHUTDOWN_MIN} minutes. Shutting down.")
                stop_event.set()
                break
            continue
        except Exception:
            time.sleep(0.1)
            continue

        print(f"\n[Real-time] Processing new data: {new_data_path}\n")
        try:
            process_new_data(new_data_path)
            trigger_retraining()
        finally:
            data_queue.task_done()
        time.sleep(0.2)

def process_new_data(new_data_path):
    """
    Process new data and make immediate predictions.
    Uses the CURRENT (possibly updated) global scaler_cluster.
    Thread-safe: protects shared `mdfcm` and `data` with locks.
    """
    global data, scaler_cluster, mdfcm
    try:
        new_data = pd.read_csv(new_data_path).copy()
        print(f"[Real-time] Loaded {len(new_data)} new records")

        X_new = new_data[cluster_cols].values
        with mdfcm_lock:
            X_new_scaled = scaler_cluster.transform(X_new)
            old_c, new_c = mdfcm.update_with_new(X_new_scaled)
            changed = (old_c != new_c)
            print(f"[Real-time] Cluster count: old={old_c}, new={new_c} ({'changed' if changed else 'no change'})")

            final_U = fcm_membership_for_points(mdfcm.X_all_, mdfcm.V, m=mdfcm.m)
            final_clusters_all = np.argmax(final_U, axis=1)

        with data_lock:
            total_rows_before = len(data)
        new_clusters = final_clusters_all[total_rows_before:] + 1
        if len(new_clusters) != len(new_data):
            print("[Real-time][WARN] Mismatch in new cluster assignment length; using last rows slice.")
        new_data['Cluster'] = new_clusters

        model_idx_path = os.path.join(output_dir, "trained_models_index.pkl")
        try:
            model_idx = joblib.load(model_idx_path)
        except Exception:
            print(f"[Real-time][WARN] Could not load model index at {model_idx_path}. Using empty index.")
            model_idx = {}

        preds_list = []
        for cluster_id in np.unique(new_clusters):
            sub = new_data[new_data['Cluster'] == cluster_id].copy()

            use_model = False
            model_path = None
            if cluster_id in model_idx:
                model_path = model_idx[cluster_id].get('model_path', None)
                if model_path and os.path.exists(model_path):
                    use_model = True

            if use_model:
                try:
                    model = joblib.load(model_path)
                    Xc = sub[features].values
                    y_pred = np.asarray(model.predict(Xc)).ravel()
                    sub['Predicted Efficiency (E)'] = y_pred
                    sub['Prediction Source'] = 'TabPFN'
                except Exception as e:
                    print(f"[Real-time][WARN] Model load/predict failed for cluster {cluster_id}: {e}. Using DEA fallback.")
                    use_model = False

            if not use_model:
                with data_lock:
                    base_in_cluster = data[data['Cluster'] == cluster_id].copy()
                combined_df = pd.concat([base_in_cluster, sub], ignore_index=True)
                dea_scores = compute_dea_scores(combined_df)
                sub_scores = dea_scores[-len(sub):]
                sub['Predicted Efficiency (E)'] = sub_scores
                sub['Prediction Source'] = 'DEA-Fallback'

            preds_list.append(sub)

        new_preds = pd.concat(preds_list, ignore_index=True) if preds_list else new_data.copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_new_pred = os.path.join(output_dir, f"new_data_predictions_{timestamp}.csv")
        new_preds.to_csv(out_new_pred, index=False)
        print(f"[Real-time] Predictions saved to: {out_new_pred}")

        with data_lock:
            data = pd.concat([data, new_data], ignore_index=True)

        return True

    except Exception as e:
        print(f"\n[Real-time] Error processing new data: {str(e)}\n")
        return False

def trigger_retraining():
    print("[Real-time] Scheduling retraining with new data")
    retrain_thread = threading.Thread(target=retrain_callback, daemon=True)
    retrain_thread.start()

# Set by main.py to call retrain_models() with correct context
retrain_callback = None