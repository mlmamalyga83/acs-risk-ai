# ============================================================
# ACS ECG Detector — SHAP analysis
# ============================================================

import numpy as np


def compute_shap_values(model, X_background, X_test, n_samples=100):
    """Вычисляет SHAP values через GradientExplainer."""
    import shap
    explainer = shap.GradientExplainer(model, X_background)
    return explainer.shap_values(X_test)


def plot_shap_beeswarm(shap_values, feature_names, save_path=None):
    """SHAP beeswarm plot."""
    import shap
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, feature_names, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
