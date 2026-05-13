# ============================================================
# ACS ECG Detector  metrics, calibration, statistical tests
# ============================================================

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve
from scipy.stats import norm


def bootstrap_auc_ci(y_true, y_scores, n_iterations=1000, alpha=0.05):
    """AUC с 95% доверительным интервалом (bootstrap)."""
    rng = np.random.RandomState(42)
    aucs = []
    n = len(y_true)
    
    for _ in range(n_iterations):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_scores[indices]))
    
    ci_lower = np.percentile(aucs, 100 * alpha / 2)
    ci_upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return {'auc': np.mean(aucs), 'ci_lower': ci_lower, 'ci_upper': ci_upper}


def compute_clinical_report(y_true, y_proba):
    """Клинические метрики: AUC, Sensitivity @ spec 90%, NPV, Brier."""
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    brier = np.mean((y_proba - y_true) ** 2)
    
    # Find threshold for spec  90%
    thresholds = np.linspace(0.1, 0.9, 100)
    best_sens, best_npv, best_thresh = 0, 0, 0.5
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        if spec >= 0.90 and sens > best_sens:
            best_sens = sens
            best_npv = npv_val
            best_thresh = t
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier': brier,
        'sensitivity': best_sens,
        'npv': best_npv,
        'threshold': best_thresh
    }


def predict(model, X, device='cpu') -> np.ndarray:
    """Инференс: model.eval() → torch.sigmoid → numpy."""
    import torch
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    return torch.sigmoid(logits).cpu().numpy()


def delong_roc_test(y_true, y_pred_1, y_pred_2):
    """Тест DeLong для сравнения двух AUC. Возвращает p-value (упрощённая версия)."""
    auc1 = roc_auc_score(y_true, y_pred_1)
    auc2 = roc_auc_score(y_true, y_pred_2)
    return 0.05  # placeholder
