# ============================================================
# ACS ECG Detector — metrics, calibration, statistical tests
# ============================================================

import numpy as np
import torch
import torch.nn as nn
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
    return {'auc': float(np.mean(aucs)), 'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper)}


def compute_clinical_report(y_true, y_proba):
    """Клинические метрики: AUC, Sensitivity @ spec 90%, NPV, Brier."""
    auc_roc = float(roc_auc_score(y_true, y_proba))
    auc_pr = float(average_precision_score(y_true, y_proba))
    brier = float(np.mean((y_proba - y_true) ** 2))

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
        'sensitivity': float(best_sens),
        'npv': float(best_npv),
        'threshold': float(best_thresh),
        'specificity': 0.90
    }


def predict(model, X, device='cpu'):
    """Инференс: model.eval() → torch.sigmoid → numpy."""
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    return torch.sigmoid(logits).cpu().numpy()


def delong_roc_test(y_true, y_pred_1, y_pred_2):
    """
    Тест ДеЛонга для сравнения двух AUC.
    Возвращает p-value.
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)

    # Вычислить AUC
    auc1 = roc_auc_score(y_true, y_pred_1)
    auc2 = roc_auc_score(y_true, y_pred_2)

    # Функция для вычисления ковариационной матрицы
    def _covariance_matrix(y_true, y_pred):
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        # Ранжировать предсказания
        order = np.argsort(y_pred)
        rank = np.argsort(order)
        rank = rank + 1  # 1-based ranks

        # V_10 для позитивных
        pos_ranks = rank[y_true == 1]
        v10 = np.mean([np.mean(rank[y_true == 0] < r) for r in pos_ranks])

        # V_01 для негативных
        neg_ranks = rank[y_true == 0]
        v01 = np.mean([np.mean(rank[y_true == 1] > r) for r in neg_ranks])

        # Ковариация
        s10 = np.var([np.mean(rank[y_true == 0] < r) for r in pos_ranks], ddof=1) if n_pos > 1 else 0
        s01 = np.var([np.mean(rank[y_true == 1] > r) for r in neg_ranks], ddof=1) if n_neg > 1 else 0

        return s10 / n_pos + s01 / n_neg

    var1 = _covariance_matrix(y_true, y_pred_1)
    var2 = _covariance_matrix(y_true, y_pred_2)

    # Ковариация между двумя моделями
    combined = np.column_stack([y_pred_1, y_pred_2])
    auc_vec = np.array([auc1, auc2])
    n_bootstrap = 500
    rng = np.random.RandomState(42)
    auc_samples = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_samples.append([roc_auc_score(y_true[idx], combined[idx, 0]),
                            roc_auc_score(y_true[idx], combined[idx, 1])])
    auc_samples = np.array(auc_samples)
    cov_matrix = np.cov(auc_samples, rowvar=False)

    z = (auc1 - auc2) / max(np.sqrt(var1 + var2 - 2 * cov_matrix[0, 1]), 1e-10)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return float(p_value)


class TemperatureScaling(nn.Module):
    """Платт-калибровка: logits / T."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


def calibrate_temperature(model, val_loader, device='cuda'):
    """Обучает temperature на val set. Возвращает (calibrator, logits_list, y_list)."""
    model.eval()
    logits_list, y_list = [], []

    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            logits_list.append(logits.cpu())
            y_list.append(batch_y)

    logits_all = torch.cat(logits_list)
    y_all = torch.cat(y_list).float()

    calibrator = TemperatureScaling()
    optimizer = torch.optim.LBFGS([calibrator.temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nn.BCEWithLogitsLoss()(calibrator(logits_all), y_all)
        loss.backward()
        return loss

    optimizer.step(closure)
    return calibrator, logits_all.numpy(), y_all.numpy()


def decision_curve_analysis(y_true, y_proba, save_path=None):
    """Decision Curve Analysis: net benefit при разных порогах."""
    thresholds = np.linspace(0, 0.5, 51)
    nb_model = []
    n = len(y_true)
    prevalence = y_true.mean()

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = tp / n - fp / n * (t / max(1 - t, 1e-10))
        nb_model.append(nb)

    nb_treat_all = [prevalence - (1 - prevalence) * t / max(1 - t, 1e-10) for t in thresholds]
    nb_treat_none = [0.0] * len(thresholds)

    # Построить график
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, nb_model, 'b-', linewidth=2, label='Model')
    ax.plot(thresholds, nb_treat_all, 'g--', linewidth=1, label='Treat all')
    ax.plot(thresholds, nb_treat_none, 'r:', linewidth=1, label='Treat none')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'thresholds': thresholds.tolist(),
        'net_benefit': nb_model,
        'max_net_benefit': float(max(nb_model))
    }


def compute_fairness_metrics(y_true, y_proba, group_masks):
    """
    Вычисляет Equal Opportunity Difference для каждой группы.
    group_masks: dict вида {'group_name': mask_array}
    Возвращает список словарей.
    """
    results = []
    for group_name, mask in group_masks.items():
        if mask.sum() < 10:
            continue
        y_g, p_g = y_true[mask], y_proba[mask]
        if len(np.unique(y_g)) < 2:
            continue
        auc = roc_auc_score(y_g, p_g)
        # Equal opportunity: TPR
        threshold = 0.5
        y_pred = (p_g >= threshold).astype(int)
        tpr = np.sum((y_pred == 1) & (y_g == 1)) / max(np.sum(y_g == 1), 1)
        results.append({
            'group': group_name,
            'size': int(mask.sum()),
            'auc': float(auc),
            'tpr': float(tpr),
            'eo_diff': None
        })

    if len(results) >= 2:
        base_tpr = results[0]['tpr']
        for r in results[1:]:
            r['eo_diff'] = abs(r['tpr'] - base_tpr)

    return results


def analyze_errors(model, X_test, y_test, patient_ids, device='cuda', n_worst=10):
    """20 worst-case ошибок: 10 FP + 10 FN с Grad-CAM."""
    probas = predict(model, X_test, device=device).ravel()

    fp_mask = (y_test == 0) & (probas >= 0.5)
    fn_mask = (y_test == 1) & (probas < 0.5)

    fp_idx = np.where(fp_mask)[0]
    fn_idx = np.where(fn_mask)[0]

    fp_worst = fp_idx[np.argsort(probas[fp_mask])[-n_worst:]] if len(fp_idx) > 0 else np.array([])
    fn_worst = fn_idx[np.argsort(probas[fn_mask])[:n_worst]] if len(fn_idx) > 0 else np.array([])

    from src.interpret.visualization import plot_ecg_with_gradcam
    from src.interpret.grad_cam import grad_cam_1d

    errors = []
    for idx in np.concatenate([fp_worst, fn_worst]) if len(fp_worst) > 0 or len(fn_worst) > 0 else []:
        idx = int(idx)
        ecg_sample = X_test[idx]
        true_label = int(y_test[idx])
        pred_prob = float(probas[idx])
        patient_id = int(patient_ids[idx])

        ecg_tensor = torch.tensor(ecg_sample, dtype=torch.float32).unsqueeze(0).to(device)
        gradcam = grad_cam_1d(model, ecg_tensor)

        save_path = f"reports/error_analysis/error_p{patient_id}_true{true_label}_pred{pred_prob:.2f}.png"
        plot_ecg_with_gradcam(ecg_sample, gradcam,
                              title=f"Patient {patient_id} | True: {true_label} | Pred: {pred_prob:.2f}",
                              save_path=save_path)

        error_type = 'FP' if true_label == 0 else 'FN'
        errors.append({
            'patient_id': patient_id,
            'type': error_type,
            'true_label': true_label,
            'pred_prob': pred_prob,
            'figure': save_path
        })

    return errors
