# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:26:48 2025

@author: kehin
"""

import os

def get_paths(output_dir):
    models_dir   = os.path.join(output_dir, "models")
    clusters_dir = os.path.join(output_dir, "clusters")
    metrics_dir  = os.path.join(output_dir, "metrics")
    plots_dir    = os.path.join(output_dir, "plots")
    shap_dir     = os.path.join(output_dir, "shap_analysis")
    return {
        "output_dir": output_dir,
        "models_dir": models_dir,
        "clusters_dir": clusters_dir,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "shap_dir": shap_dir,
    }

def ensure_dirs(paths_dict):
    for p in paths_dict.values():
        os.makedirs(p, exist_ok=True)