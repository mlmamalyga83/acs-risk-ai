# ============================================================
# ACS ECG Detector — baseline classifiers
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd


def train_baseline_models(X, y, patient_ids, cv_folds=5):
    """Обучает LogisticRegression, RandomForest, XGBoost с GroupKFold."""
    kf = GroupKFold(n_splits=cv_folds)
    
    results = {}
    
    # LogisticRegression (with StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'logistic': LogisticRegression(C=0.1, max_iter=2000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  class_weight='balanced', random_state=42),
    }
    
    for name, model in models.items():
        fold_aucs = []
        X_use = X_scaled if name == 'logistic' else X
        
        for train_idx, val_idx in kf.split(X, y, groups=patient_ids):
            model.fit(X_use[train_idx], y[train_idx])
            probas = model.predict_proba(X_use[val_idx])[:, 1]
            from sklearn.metrics import roc_auc_score
            fold_aucs.append(roc_auc_score(y[val_idx], probas))
        
        results[name] = {'auc_mean': np.mean(fold_aucs), 'auc_std': np.std(fold_aucs)}
        print(f"  {name}: AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    
    return results
