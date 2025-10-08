# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:11:16 2025

@author: kehin
"""

from tabpfn import TabPFNRegressor

def build_and_fit_tabpfn(X_train, y_train, device="cpu"):
    model = TabPFNRegressor(device=device)
    model.fit(X_train, y_train.ravel())
    return model