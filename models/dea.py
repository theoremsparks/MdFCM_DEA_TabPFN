# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:10:17 2025

@author: kehin
"""

import numpy as np
import pulp

def compute_dea_scores(cluster_df):
    """
    Compute DDF-DEA-style efficiency scores for all rows in cluster_df.
    """
    inputs_mixed = cluster_df[['Current Ratio (C/R)', 'Asset Turnover (A/T)', 'Debt to Asset (D/A)']].values
    good_outputs = cluster_df['Return (R)'].values
    bad_outputs  = cluster_df['Risk (σ)'].values
    
    R_positive = good_outputs.clip(min=0)
    R_negative = (-good_outputs).clip(min=0)

    input_positive_parts, input_negative_parts = [], []
    for k in range(inputs_mixed.shape[1]):
        input_positive_parts.append(inputs_mixed[:, k].clip(min=0))
        input_negative_parts.append((-inputs_mixed[:, k]).clip(min=0))

    n = len(cluster_df)
    scores = []

    for i in range(n):
        prob = pulp.LpProblem(f"DEA_Inefficiency_{i+1}", pulp.LpMaximize)
        e = pulp.LpVariable('e', lowBound=0)
        a = [pulp.LpVariable(f'a_{j+1}', lowBound=0) for j in range(n)]
        prob += e

        for k in range(inputs_mixed.shape[1]):
            prob += pulp.lpSum(input_positive_parts[k][j] * a[j] for j in range(n)) <= input_positive_parts[k][i]
            prob += pulp.lpSum(input_negative_parts[k][j] * a[j] for j in range(n)) >= input_negative_parts[k][i]

        prob += pulp.lpSum(R_positive[j] * a[j] for j in range(n)) >= R_positive[i] * (1 + e)
        prob += pulp.lpSum(R_negative[j] * a[j] for j in range(n)) <= R_negative[i] * (1 - e)
        prob += pulp.lpSum(bad_outputs[j] * a[j] for j in range(n)) == bad_outputs[i] * (1 - e)
        prob += pulp.lpSum(a[j] for j in range(n)) == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        ineff = pulp.value(e)
        eff = None if ineff is None else (1 - ineff) / (1 + ineff)
        scores.append(eff)
    return np.array(scores, dtype=float)