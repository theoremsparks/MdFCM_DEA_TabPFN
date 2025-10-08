# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:19:07 2025

@author: kehin
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tabpfn_extensions import interpretability

def perform_shap_analysis(model, X_data, feature_names, cluster_id, output_dir):
    display_cluster_id = cluster_id + 1
    print(f"  Running SHAP analysis for cluster {display_cluster_id}...")
    shap_values = interpretability.shap.get_shap_values(estimator=model, test_x=X_data,
                                                        attribute_names=feature_names, algorithm="permutation")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = interpretability.shap.plot_shap(shap_values)
    plt.title(f"Cluster {display_cluster_id} SHAP Analysis", fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"cluster_{display_cluster_id}_shap_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  SHAP plot saved to: {plot_path}")
    
    shap_df = pd.DataFrame(shap_values.values, columns=[f"SHAP_{name}" for name in feature_names])
    csv_path = os.path.join(output_dir, f"cluster_{display_cluster_id}_shap_values_{timestamp}.csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"  SHAP values saved to CSV\n")
    
    print(f"  SHAP analysis for cluster {display_cluster_id} completed!")