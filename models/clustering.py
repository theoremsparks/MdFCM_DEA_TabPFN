# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:08:11 2025

@author: kehin
"""

import numpy as np
from sklearn.metrics import pairwise_distances

# ====== Clustering utilities (FCM / MdFCM) ======
def xie_beni_index_XU(X, U, V, m=2):
    dist2 = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2) ** 2
    num = np.sum((U ** m) * dist2)
    diffs = V[:, None, :] - V[None, :, :]
    mask = ~np.eye(V.shape[0], dtype=bool)
    min_csep = np.min(np.sum(diffs**2, axis=2)[mask])
    return num / (X.shape[0] * (min_csep + 1e-12))

def fcm_once(X, c, m=2, max_iter=300, tol=1e-5, seed=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    V = X[rng.choice(n, size=c, replace=False)]
    for _ in range(max_iter):
        dist = np.linalg.norm(X[:, None, :] - V[None, :], axis=2) + 1e-12
        inv = 1.0 / dist
        inv_pow = inv ** (2/(m-1))
        U = inv_pow / np.sum(inv_pow, axis=1, keepdims=True)
        Um = U ** m
        V_new = (Um.T @ X) / (np.sum(Um, axis=0)[:, None] + 1e-12)
        if np.linalg.norm(V_new - V) < tol:
            V = V_new
            break
        V = V_new
    return U, V

def fcm_membership_for_points(X_pts, V, m=2):
    dist = np.linalg.norm(X_pts[:, None, :] - V[None, :, :], axis=2) + 1e-12
    inv = 1.0 / dist
    inv_pow = inv ** (2/(m-1))
    U = inv_pow / np.sum(inv_pow, axis=1, keepdims=True)
    return U

class MdFCM:
    """
    Modified dFCM:
      - Trigger uses ONLY the new batch vs min inter-center spacing.
      - If no trigger: movement with fixed c on [old + new].
      - If trigger: evaluate k in {c-1, c, c+1} via XB on [old + new]; choose best.
    """
    def __init__(self, c_init=5, m=2, max_iter=300, tol=1e-5, seed=42):
        self.c = c_init
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.U = None
        self.V = None
        self.X_all_ = None

    def fit(self, X_init):
        X_init = np.asarray(X_init)
        self.X_all_ = np.array(X_init, copy=True)
        self.U, self.V = fcm_once(self.X_all_, self.c, m=self.m,
                                  max_iter=self.max_iter, tol=self.tol, seed=self.seed)
        return self

    def _min_center_distance(self):
        if self.V is None or self.V.shape[0] < 2:
            return np.inf
        D = pairwise_distances(self.V)
        np.fill_diagonal(D, np.inf)
        return float(np.min(D))

    def _evaluate_k(self, X, k):
        U_k, V_k = fcm_once(X, k, m=self.m, max_iter=self.max_iter, tol=self.tol, seed=self.seed)
        xb = xie_beni_index_XU(X, U_k, V_k, m=self.m)
        return xb, U_k, V_k

    def update_with_new(self, X_new):
        X_new = np.asarray(X_new)
        if X_new.size == 0:
            return self.c, self.c

        if self.V is None or self.X_all_ is None:
            self.fit(X_new)
            return self.c, self.c

        # Trigger from new batch only
        min_sep = self._min_center_distance()
        d_new_min = np.linalg.norm(X_new[:, None, :] - self.V[None, :, :], axis=2).min(axis=1)
        trigger = np.any(d_new_min > min_sep)

        X_all = np.vstack([self.X_all_, X_new])
        old_c = self.c

        if not trigger:
            xb, U_k, V_k = self._evaluate_k(X_all, self.c)
            self.U, self.V = U_k, V_k
            self.X_all_ = X_all
            return old_c, self.c

        if self.c > 1:
            candidates = [self.c - 1, self.c, self.c + 1]
        else:
            candidates = [self.c, self.c + 1]

        best = None
        for k in candidates:
            xb, U_k, V_k = self._evaluate_k(X_all, k)
            if best is None or xb < best[0]:
                best = (xb, k, U_k, V_k)

        xb, k_best, U_best, V_best = best
        self.c, self.U, self.V = k_best, U_best, V_best
        self.X_all_ = X_all
        return old_c, self.c