# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:29:00 2025

@author: kehin
"""

import os
import time
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
)

from models import (
    MdFCM, fcm_once, xie_beni_index_XU,
    compute_dea_scores,
    build_and_fit_tabpfn
)
from metrics import smape, mean_error, normalized_root_mean_squared_error
from plotting import plot_scatter_with_fit, perform_shap_analysis
from retraining import (
    real_time_prediction_worker, process_new_data, trigger_retraining,
    retrain_models, watch_for_new_data
)
from retraining import realtime as realtime_mod
from retraining import retrain as retrain_mod
from utils import get_paths, ensure_dirs, get_sync_objects

def main():
    # ====== System up time ======
    run_start_dt = datetime.now()
    print(f"[RUN] Started at {run_start_dt:%Y-%m-%d %H:%M:%S}")

    # ====== Reproducibility ======
    seed = 42

    # ====== Device & Paths ======
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    output_dir = "C:/Users/kehin/Documents/Spyder/MdFCM_DEA_TabPFN_Real_Time/results"
    paths = get_paths(output_dir)
    ensure_dirs(paths)

    models_dir   = paths["models_dir"]
    clusters_dir = paths["clusters_dir"]
    metrics_dir  = paths["metrics_dir"]
    plots_dir    = paths["plots_dir"]
    shap_dir     = paths["shap_dir"]

    # ====== Sync objects ======
    sync = get_sync_objects()
    data_queue = sync["data_queue"]
    data_lock = sync["data_lock"]
    mdfcm_lock = sync["mdfcm_lock"]
    stop_event = sync["stop_event"]

    # ====== Load Base Data ======
    data_path = "C:/Users/kehin/Documents/Spyder/MdFCM_DEA_TabPFN_Real_Time/demo_data/data.csv"
    data = pd.read_csv(data_path)

    # ====== Feature columns ======
    cluster_cols = ["Current Ratio (C/R)", "Asset Turnover (A/T)", "Debt to Asset (D/A)", "Return (R)", "Risk (σ)"]
    features     = cluster_cols[:]  # predictors for TabPFN
    feature_names = features

    # ====== Scale, choose initial c, fit MdFCM on BASE data ======
    X_cluster = data[cluster_cols].values
    scaler_cluster = StandardScaler().fit(X_cluster)
    X_scaled = scaler_cluster.transform(X_cluster)

    best_xb, best_k, best_U_init, best_V_init = float('inf'), None, None, None
    for k in range(3, 10):
        U_k, V_k = fcm_once(X_scaled, k, m=2, max_iter=300, tol=1e-5, seed=seed)
        xb_k = xie_beni_index_XU(X_scaled, U_k, V_k, m=2)
        if xb_k < best_xb:
            best_xb, best_k, best_U_init, best_V_init = xb_k, k, U_k, V_k

    print(f"Initial optimal number of clusters (by XB): {best_k}")

    mdfcm = MdFCM(c_init=best_k, m=2, max_iter=300, tol=1e-5, seed=seed)
    mdfcm.U, mdfcm.V = best_U_init, best_V_init
    mdfcm = mdfcm.fit(X_scaled)

    centers_scaled = mdfcm.V
    centers_orig = scaler_cluster.inverse_transform(centers_scaled)
    centroids_df = pd.DataFrame(centers_orig, columns=cluster_cols)
    centroids_df.insert(0, "Cluster", range(1, mdfcm.c + 1))
    print("\nCluster centroids - BASE:")
    print(centroids_df)
    centroids_df.to_csv(os.path.join(output_dir, "cluster_centroids_BASE.csv"), index=False)

    base_clusters = np.argmax(mdfcm.U, axis=1)
    data = data.copy()
    data['Cluster'] = base_clusters + 1

    all_results = []
    trained_models = {}

    # ====== Train TabPFN per cluster (using DEA labels) & Save artifacts ======
    for cluster_id in range(mdfcm.c):
        display_cluster_id = cluster_id + 1
        cluster_df = data[data['Cluster'] == display_cluster_id].copy()
        n_cluster = len(cluster_df)
        print(f"\nCluster {display_cluster_id}: n={n_cluster}")

        if n_cluster < 20:
            print(f"  Skipping training (too small).")
            continue

        eff_scores = compute_dea_scores(cluster_df)
        cluster_df['Efficiency Score (E)'] = eff_scores
        cluster_df = cluster_df.dropna(subset=['Efficiency Score (E)']).reset_index(drop=True)

        cluster_csv = os.path.join(clusters_dir, f"cluster_{display_cluster_id}.csv")
        cluster_df.to_csv(cluster_csv, index=False)

        X = cluster_df[features].values
        y = cluster_df['Efficiency Score (E)'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=seed)

        tabpfn_model = build_and_fit_tabpfn(X_train, y_train, device=device)

        y_train_pred = tabpfn_model.predict(X_train).reshape(-1, 1)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        train_smape = smape(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_r = np.corrcoef(y_train.flatten(), y_train_pred.flatten())[0, 1]
        train_me = mean_error(y_train, y_train_pred)
        train_nrmse = normalized_root_mean_squared_error(y_train, y_train_pred)

        y_test_pred = tabpfn_model.predict(X_test).reshape(-1, 1)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        test_smape = smape(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_r = np.corrcoef(y_test.flatten(), y_test_pred.flatten())[0, 1]
        test_me = mean_error(y_test, y_test_pred)
        test_nrmse = normalized_root_mean_squared_error(y_test, y_test_pred)

        plot_scatter_with_fit(y_train.flatten(), y_train_pred.flatten(), "Train", cluster_id, plots_dir)
        plot_scatter_with_fit(y_test.flatten(), y_test_pred.flatten(), "Test", cluster_id, plots_dir)

        perform_shap_analysis(tabpfn_model, X_test, feature_names, cluster_id, shap_dir)

        readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.join(metrics_dir, f"cluster_{display_cluster_id}_metrics.txt")
        
        with open(filename, 'a') as file:
            file.write("\n" + "="*50 + "\n")
            file.write(f"Run Timestamp: {readable_time}\n")
            file.write(f"Cluster ID: {display_cluster_id}\n")
            file.write(f"Training samples: {len(X_train)}\n")
            file.write(f"Test samples: {len(X_test)}\n")
            
            file.write("\n=== Training Data Metrics ===\n")
            file.write(f"MSE       : {train_mse:.4f}\n")
            file.write(f"RMSE      : {train_rmse:.4f}\n")
            file.write(f"MAE       : {train_mae:.4f}\n")
            file.write(f"MAPE      : {train_mape:.2f}%\n")
            file.write(f"sMAPE     : {train_smape:.2f}%\n")
            file.write(f"R²        : {train_r2:.4f}\n")
            file.write(f"R         : {train_r:.4f}\n")
            file.write(f"ME        : {train_me:.4f}\n")
            file.write(f"NRMSE     : {train_nrmse:.4f}\n") 

            file.write("\n=== Testing Data Metrics ===\n")
            file.write(f"MSE       : {test_mse:.4f}\n")
            file.write(f"RMSE      : {test_rmse:.4f}\n")
            file.write(f"MAE       : {test_mae:.4f}\n")
            file.write(f"MAPE      : {test_mape:.2f}%\n")
            file.write(f"sMAPE     : {test_smape:.2f}%\n")
            file.write(f"R²        : {test_r2:.4f}\n")
            file.write(f"R         : {test_r:.4f}\n")
            file.write(f"ME        : {test_me:.4f}\n")
            file.write(f"NRMSE     : {test_nrmse:.4f}\n") 

        predictions_df = pd.DataFrame({
            "Actual": y_test.flatten(), 
            "Predicted": y_test_pred.flatten()
        })
        predictions_df.to_csv(os.path.join(metrics_dir, f"cluster_{display_cluster_id}_test_predictions.csv"), index=False)

        metrics = {'Cluster': display_cluster_id, 'Train_RMSE': float(train_rmse), 'Train_MAE': float(train_mae), 'Train_MAPE': float(train_mape),
                   'Train_sMAPE': float(train_smape), 'Train_R2': float(train_r2), 'Train_R': float(train_r), 'Train_ME': float(train_me), 
                   'Train_NRMSE': float(train_nrmse), 'Test_RMSE': float(test_rmse), 'Test_MAE': float(test_mae), 'Test_MAPE': float(test_mape),
                   'Test_sMAPE': float(test_smape), 'Test_R2': float(test_r2), 'Test_R': float(test_r), 'Test_ME': float(test_me),
                   'Test_NRMSE': float(test_nrmse), 'n_train': int(len(X_train)), 'n_test': int(len(X_test))}
        all_results.append(metrics)

        joblib.dump(tabpfn_model, os.path.join(models_dir, f"tabpfn_cluster_{display_cluster_id}.joblib"))

        trained_models[display_cluster_id] = {'model_path': os.path.join(models_dir, f"tabpfn_cluster_{display_cluster_id}.joblib"), 
                                              'n_train': int(len(X_train))}

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "cluster_metrics.csv"), index=False)
    joblib.dump(scaler_cluster, os.path.join(output_dir, "scaler_cluster.joblib"))
    joblib.dump(mdfcm,          os.path.join(output_dir, "mdfcm.pkl"))
    joblib.dump(trained_models, os.path.join(output_dir, "trained_models_index.pkl"))
    print("\n====== Training complete. Artifacts saved. ======\n")

    # ====== Initialize retraining modules with shared state ======
    realtime_shared = {
        "data": data,
        "scaler_cluster": scaler_cluster,
        "mdfcm": mdfcm,
        "output_dir": output_dir,
        "features": features,
        "cluster_cols": cluster_cols,
        "data_queue": data_queue,
        "data_lock": data_lock,
        "mdfcm_lock": mdfcm_lock,
        "stop_event": stop_event,
        "IDLE_SHUTDOWN_MIN": 10,
        "QUEUE_TIMEOUT_SEC": 300,
    }
    realtime_mod.init_runtime(realtime_shared)
    retrain_mod.init_runtime({
        "data": data,
        "mdfcm": mdfcm,
        "scaler_cluster": scaler_cluster,
        "data_lock": data_lock,
        "mdfcm_lock": mdfcm_lock,
        "output_dir": output_dir,
        "models_dir": models_dir,
        "features": features,
        "cluster_cols": cluster_cols,
        "seed": seed,
        "device": device,
    })

    # connect retrain callback
    def retrain_callback():
        retrain_mod.retrain_models()
    realtime_mod.retrain_callback = retrain_callback

    prediction_thread = threading.Thread(target=realtime_mod.real_time_prediction_worker, daemon=True)
    prediction_thread.start()

    data_directory = "C:/Users/kehin/Documents/Spyder/MdFCM_DEA_TabPFN_Real_Time/demo_data"
    watcher_thread = threading.Thread(
        target=watch_for_new_data,
        args=(data_directory,),
        kwargs={"data_queue": data_queue, "stop_event": stop_event},
        daemon=True
    )
    watcher_thread.start()

    print("\n[System] Real-time adaptive system is now active\n")
    print("\n[System] Waiting for new data....\n")

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[System] Shutting down real-time system\n")
        stop_event.set()

    prediction_thread.join(timeout=5)
    watcher_thread.join(timeout=5)
    print("\n[System] All threads exited cleanly....\n")

    run_end_dt = datetime.now()
    elapsed = run_end_dt - run_start_dt
    print(f"[RUN] Ended at   {run_end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"[RUN] Elapsed    {elapsed}  (hh:mm:ss)")

    training_time_file = os.path.join(output_dir, "training_time.txt")
    with open(training_time_file, 'a') as f:
        f.write(f"Training Run Information\n")
        f.write(f"=========================================\n\n")
        f.write(f"Start Time: {run_start_dt:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"End Time:   {run_end_dt:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Elapsed Time: {elapsed}\n")
        f.write(f"Total Hours: {elapsed.total_seconds() / 3600:.4f}\n")
        f.write(f"Total Minutes: {elapsed.total_seconds() / 60:.2f}\n")
        f.write(f"Total Seconds: {elapsed.total_seconds():.2f}\n\n")
        f.write(f"Number of Clusters: {mdfcm.c}\n")
        f.write(f"Total Training Samples: {len(data)}\n")

    print(f"System Uptime details saved to: {training_time_file}")

if __name__ == "__main__":
    import threading  # keep local to main to mirror your original behavior
    main()