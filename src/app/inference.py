# ============================================================
# ACS ECG Detector — inference pipeline
# Единая точка входа для Streamlit
# signal: [samples, 12] — сырой сигнал в mV
# clinical: {'age': 62, 'sex': 'M'}
# ============================================================

import numpy as np
import torch
from pathlib import Path


def load_model(model_path="models/resnet1d_encoder.pt", device="cpu"):
    """Загружает модель."""
    from src.models.cnn_model import ResNet1D

    model = ResNet1D(dropout=0.3)
    ckpt = torch.load(model_path, map_location=device)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


def preprocess_ecg_for_inference(signal, fs=500):
    """Предобработка ЭКГ: фильтр, R-пики, сегментация, нормализация.
    Возвращает: [n_cycles, 12, 350] или None при ошибке."""
    from src.preprocessing.filters import preprocess_ecg_signal
    from src.preprocessing.segmentation import extract_heartbeats, segment_all_leads

    signal = preprocess_ecg_signal(signal, fs)
    signal[:, 3] *= -1.0  # aVR invert

    lead_energy = [np.std(signal[:, ch]) for ch in range(min(4, signal.shape[1]))]
    best_lead = np.argmax(lead_energy) if lead_energy else 1

    beats = extract_heartbeats(signal, fs, lead_idx=best_lead)
    if len(beats['r_peaks']) < 3:
        return None

    cycles = segment_all_leads(signal, fs, beats['r_peaks'])
    if len(cycles) == 0:
        return None

    for c in range(len(cycles)):
        mean_vals = np.mean(cycles[c], axis=0, keepdims=True)
        std_vals = np.std(cycles[c], axis=0, keepdims=True)
        cycles[c] = (cycles[c] - mean_vals) / np.maximum(std_vals, 1e-8)

    return np.transpose(cycles, (0, 2, 1)).astype(np.float32)


def predict_with_uncertainty(model, cycles, n_samples=50, device="cpu"):
    """MC Dropout: среднее + 95% CI."""
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.apply(enable_dropout)
    all_probas = []

    with torch.no_grad():
        for _ in range(n_samples):
            batch = torch.tensor(cycles, dtype=torch.float32).to(device)
            out = torch.sigmoid(model(batch))
            all_probas.append(out.cpu().numpy())

    model.eval()
    all_probas = np.array(all_probas)
    mean_proba = np.mean(all_probas, axis=0)
    mean_patient = float(np.mean(mean_proba))
    ci = np.percentile(np.mean(all_probas, axis=1), [2.5, 97.5])

    return {
        'mean': mean_patient,
        'ci_95': [float(ci[0]), float(ci[1])],
        'per_cycle': mean_proba
    }


def estimate_heart_rate(r_peaks, fs=500):
    """ЧСС по R-R интервалам."""
    if r_peaks is None or len(r_peaks) < 2:
        return 75
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    return int(60.0 / mean_rr) if mean_rr > 0 else 75


def run_inference(signal, clinical, model=None, model_path="models/resnet1d_encoder.pt", device="cpu"):
    """
    Полный пайплайн: от сигнала до результата.

    Args:
        signal: np.ndarray [samples, 12]
        clinical: dict {'age': 62, 'sex': 'M'}
        model: nn.Module или None (загрузит из model_path)
        model_path: путь к .pt файлу
        device: 'cpu' или 'cuda'

    Returns:
        dict со всеми результатами для UI
    """
    from src.features.clinical_features import preprocess_clinical
    from src.interpret.grad_cam import grad_cam_1d
    from src.interpret.visualization import segment_importance_table
    from src.app.report_generator import generate_auto_report
    from src.app.red_flags import check_red_flags

    if model is None:
        model = load_model(model_path, device)

    # Preprocess
    cycles = preprocess_ecg_for_inference(signal)
    if cycles is None:
        return {'error': 'Не удалось обработать ЭКГ: не найдены R-пики'}

    # Model inference with MC Dropout
    uncertainty = predict_with_uncertainty(model, cycles, n_samples=30, device=device)
    risk_score = uncertainty['mean']
    risk_ci = uncertainty['ci_95']

    # Risk category based on context
    def get_risk_category(score, context="Приёмное отделение"):
        thresholds = {'Поликлиника': (0.20, 0.50),
                      'Приёмное отделение': (0.15, 0.40),
                      'Стационар': (0.10, 0.30)}
        low, high = thresholds.get(context, (0.15, 0.40))
        if score <= low:
            return "НИЗКИЙ РИСК"
        elif score <= high:
            return "УМЕРЕННЫЙ РИСК"
        return "ВЫСОКИЙ РИСК"

    risk_category = get_risk_category(risk_score, clinical.get('context', 'Приёмное отделение'))

    # Grad-CAM (first cycle)
    cycle_tensor = torch.tensor(cycles[0:1], dtype=torch.float32).to(device)
    gradcam_map = grad_cam_1d(model, cycle_tensor)
    seg_table = segment_importance_table(gradcam_map)

    # Heart rate
    hr = estimate_heart_rate(None)

    # Red flags
    red_flags = check_red_flags(signal)

    # Signal quality
    signal_quality = "Хорошее" if np.std(signal) > 1e-5 else "Плохое"

    # Auto report
    auto_report = generate_auto_report(risk_score, gradcam_map, hr, "Синусовый", red_flags)

    return {
        'risk_score': risk_score,
        'risk_ci': risk_ci,
        'risk_category': risk_category,
        'gradcam_map': gradcam_map,
        'segment_table': seg_table,
        'heart_rate': hr,
        'red_flags': red_flags,
        'signal_quality': signal_quality,
        'auto_report': auto_report,
        'n_cycles': len(cycles),
    }
