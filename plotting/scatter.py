# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:17:41 2025

@author: kehin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_scatter_with_fit(y_true, y_pred, dataset_type: str, cluster_id: int, output_dir: str):
    # Convert cluster_id to 1-based for display
    display_cluster_id = cluster_id + 1
    
    # Best-fit line
    a, b = np.polyfit(y_true, y_pred, 1)
    
    # Create x values that include the origin and full range of data
    x_min = min(np.min(y_true), 0)
    x_max = max(np.max(y_true), 0)
    x_line = np.linspace(x_min, x_max, 200)
    y_line = a * x_line + b

    # Pearson correlation coefficient
    R = np.corrcoef(y_true, y_pred)[0, 1]

    plt.figure(figsize=(7.2, 6.4))
    plt.scatter(y_true, y_pred, alpha=0.65, edgecolor='k', linewidth=0.5)

    ideal_x = np.array([x_min, x_max])
    plt.plot(ideal_x, ideal_x, linestyle='--', linewidth=2, color='red', label='Ideal: y = x')

    plt.plot(x_line, y_line, linewidth=2, color='blue', label=f'Best fit: y = {a:.3f}x + {b:.3f}')

    plt.title(f'Cluster {display_cluster_id} {dataset_type} Set: Predicted vs Actual (TabPFN)', fontsize=12)
    plt.xlabel('Actual Efficiency Score', fontsize=12)
    plt.ylabel('Predicted Efficiency Score', fontsize=12)
    plt.legend(fontsize=12, loc='lower right')

    eq_text = f'R = {R:.4f}'

    plt.gca().text(0.02, 0.98, eq_text, transform=plt.gca().transAxes, ha='left', va='top', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.8, linewidth=0.5))

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"cluster_{display_cluster_id}_scatter_{dataset_type.lower()}_{timestamp}.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {outpath}")