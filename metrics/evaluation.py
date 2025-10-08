# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:15:48 2025

@author: kehin
"""

import numpy as np
from sklearn.metrics import mean_squared_error

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_true)))

def mean_error(y_true, y_pred):
    return np.mean(np.array(y_pred) - np.array(y_true))

def normalized_root_mean_squared_error(y_true, y_pred):
    """
    Calculate Normalized Root Mean Square Error (NRMSE) using range normalization
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true = np.array(y_true)
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range != 0 else float('inf')
    return nrmse