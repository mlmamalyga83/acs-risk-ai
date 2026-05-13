# ============================================================
# ACS ECG Detector — baseline classifiers (Stage 3)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def train_baseline_models(X, y, patient_ids, cv_folds=5):
    """Trains LogisticRegression, RandomForest, XGBoost with GroupKFold."""
    kf = GroupKFold(n_splits=cv_folds)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'logistic': LogisticRegression(C=0.1, max_iter=2000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                                 class_weight='balanced', random_state=42,
                                                 n_jobs=-1),
        'xgboost': None,
    }
    
    # XGBoost — try to import
    try:
        from xgboost import XGBClassifier
        scale_pos_weight = max(len(y) - y.sum(), 1) / max(y.sum(), 1)
        models['xgboost'] = XGBClassifier(n_estimators=100, max_depth=5,
                                            scale_pos_weight=scale_pos_weight,
                                            random_state=42, eval_metric='logloss')
    except ImportError:
        print("  XGBoost not installed — skipping")
        del models['xgboost']
    
    results = {}
    for name, model in models.items():
        if model is None:
            continue
        
        fold_aucs = []
        X_use = X_scaled if name == 'logistic' else X
        
        for train_idx, val_idx in kf.split(X, y, groups=patient_ids):
            model.fit(X_use[train_idx], y[train_idx])
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_use[val_idx])[:, 1]
            else:
                probas = model.predict(X_use[val_idx])
            fold_aucs.append(roc_auc_score(y[val_idx], probas))
        
        results[name] = {
            'auc_mean': float(np.mean(fold_aucs)),
            'auc_std': float(np.std(fold_aucs)),
            'model': model
        }
        print(f"  {name}: AUC = {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")
    
    return results


def find_thresholds(y_true, y_proba):
    """Find optimal thresholds by 3 criteria."""
    from sklearn.metrics import f1_score, recall_score
    
    thresholds = np.linspace(0.1, 0.9, 80)
    best = {'f1': (0, 0.5), 'sensitivity95': (0, 0.5), 'youden': (0, 0.5)}
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn = (y_true[y_pred == 0] == 0).sum()
        fp = (y_true[y_pred == 0] == 1).sum()
        fn = (y_true[y_pred == 1] == 0).sum()
        tp = (y_true[y_pred == 1] == 1).sum()
        
        # F1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best['f1'][0]:
            best['f1'] = (f1, t)
        
        # Sensitivity at spec >= 95%
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if spec >= 0.95 and sens > best['sensitivity95'][0]:
            best['sensitivity95'] = (sens, t)
        
        # Youden index
        youden = sens + spec - 1
        if youden > best['youden'][0]:
            best['youden'] = (youden, t)
    
    return {k: v[1] for k, v in best.items()}


def run_baseline_stage(config_path: str = "config/config.yaml"):
    """Stage 3: train baseline models on hand-crafted features."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.features.ecg_features import extract_ecg_features
    from src.features.heart_score import compute_heart_score
    
    print("=" * 40)
    print("Stage 3: Baseline Models")
    print("=" * 40)
    
    # ---- Load data ----
    processed = Path('data/processed')
    
    # Train data — batch files
    manifest = processed / 'X_train_manifest.txt'
    if manifest.exists():
        with open(manifest) as f:
            batch_files = [line.strip() for line in f if line.strip()]
        X_train_parts = [np.load(b) for b in batch_files]
        X_train = np.concatenate(X_train_parts, axis=0)
        print(f"Loaded train from {len(batch_files)} batch files: {X_train.shape}")
    else:
        X_train = np.load(processed / 'X_train.npy')
        print(f"Loaded train: {X_train.shape}")
    
    X_val = np.load(processed / 'X_val.npy')
    y_train = np.load(processed / 'y_train.npy')
    y_val = np.load(processed / 'y_val.npy')
    pids_train = np.load(processed / 'patient_ids_train.npy')
    pids_val = np.load(processed / 'patient_ids_val.npy')
    clinical_train = np.load(processed / 'clinical_train.npy')
    clinical_val = np.load(processed / 'clinical_val.npy')
    
    print(f"Train: {len(X_train)} cycles, ACS={y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"Val:   {len(X_val)} cycles, ACS={y_val.sum()}/{len(y_val)} ({y_val.mean()*100:.1f}%)")
    
    # ---- Extract features ----
    # Process in batches to avoid OOM (100K x 12 x 350 = 1.56 GB)
    n_train_sample = min(len(X_train), 50000)
    train_idx_sample = np.random.RandomState(42).choice(len(X_train), n_train_sample, replace=False)
    
    print(f"\nExtracting features from {n_train_sample} train cycles (in batches)...")
    df_parts = []
    batch_size = 10000
    for start in range(0, n_train_sample, batch_size):
        end = min(start + batch_size, n_train_sample)
        batch_idx = train_idx_sample[start:end]
        df_part = extract_ecg_features(X_train[batch_idx])
        df_part['age'] = clinical_train[batch_idx, 0]
        df_part['sex'] = clinical_train[batch_idx, 1]
        df_parts.append(df_part)
    df_train = pd.concat(df_parts, ignore_index=True)
    
    print(f"Extracting features from {len(X_val)} val cycles...")
    df_val = extract_ecg_features(X_val)
    df_val['age'] = clinical_val[:, 0]
    df_val['sex'] = clinical_val[:, 1]
    
    X_train_feat = df_train.values.astype(np.float32)
    X_val_feat = df_val.values.astype(np.float32)
    y_train_sample = y_train[train_idx_sample]
    pids_train_sample = pids_train[train_idx_sample]
    
    n_feat = X_train_feat.shape[1]
    print(f"Features: {n_feat} (89 ECG + 2 clinical)")
    
    # Handle NaN/Inf
    X_train_feat = np.nan_to_num(X_train_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_feat = np.nan_to_num(X_val_feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ---- Train ----
    print("\nTraining baseline models...")
    results = train_baseline_models(X_train_feat, y_train_sample, pids_train_sample)
    
    # Best model
    best_name = max(results, key=lambda n: results[n]['auc_mean'])
    best_model = results[best_name]['model']
    best_auc_cv = results[best_name]['auc_mean']
    print(f"\nBest model: {best_name} (CV AUC = {best_auc_cv:.3f})")
    
    # ---- Thresholds ----
    print("\nFinding optimal thresholds...")
    scaler = StandardScaler() if best_name == 'logistic' else None
    X_val_use = scaler.fit_transform(X_val_feat) if scaler else X_val_feat
    
    if hasattr(best_model, 'predict_proba'):
        val_probas = best_model.predict_proba(X_val_use)[:, 1]
    else:
        val_probas = best_model.predict(X_val_use)
    
    thresholds = find_thresholds(y_val, val_probas)
    for k, v in thresholds.items():
        print(f"  {k}: threshold = {v:.2f}")
    
    # ---- HEART comparison ----
    print("\nHEART-score comparison...")
    val_ages = clinical_val[:, 0]
    
    heart_b = compute_heart_score(val_ages, np.ones(len(val_ages)), [])
    
    auc_model = roc_auc_score(y_val, val_probas)
    auc_heart_b = roc_auc_score(y_val, heart_b)
    
    print(f"  Model (val AUC):        {auc_model:.3f}")
    print(f"  HEART B (honest):       {auc_heart_b:.3f}")
    print(f"  Delta (model vs HEART):  {auc_model - auc_heart_b:+.3f}")
    
    # ---- SHAP ----
    print("\nComputing SHAP values...")
    _run_shap_analysis(best_model, X_val_feat, df_val.columns, n_background=100, n_test=500)
    
    # ---- Save ----
    Path('models').mkdir(exist_ok=True)
    import joblib
    joblib.dump(best_model, 'models/baseline_best.pkl')
    print(f"OK Baseline model saved: models/baseline_best.pkl")
    
    print(f"\n{'=' * 40}")
    print("OK Stage 3 complete")
    print(f"  Best: {best_name} (CV AUC = {best_auc_cv:.3f})")
    print(f"  Val AUC: {auc_model:.3f}")
    print(f"{'=' * 40}")
    
    return results


def _run_shap_analysis(model, X, feature_names, n_background=100, n_test=500):
    """SHAP beeswarm + bar plots for top features."""
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        Path('reports/figures').mkdir(parents=True, exist_ok=True)
        
        # Subsample
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(len(X), min(n_background, len(X)), replace=False)
        test_idx = rng.choice(len(X), min(n_test, len(X)), replace=False)
        
        X_bg = X[bg_idx]
        X_test = X[test_idx]
        
        # Use TreeExplainer for tree models, otherwise KernelExplainer
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_bg[:50])
        
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        
        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type='bar', show=False, max_display=15)
        fig.savefig('reports/figures/shap_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Beeswarm
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          show=False, max_display=15)
        fig.savefig('reports/figures/shap_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  SHAP plots saved: reports/figures/shap_*.png")
    except Exception as e:
        print(f"  WARN SHAP failed: {e}")
