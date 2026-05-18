# ============================================================
# ACS ECG Detector — inference pipeline
# Единая точка входа для Streamlit
# signal: [samples, 12] — сырой сигнал в mV
# clinical: {'age': 62, 'sex': 'M'}
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


RHYTHM_CLASSES = {0: "Синусовый ритм", 1: "Фибрилляция/трепетание предсердий", 2: "Другой ритм"}


def load_rhythm_model(model_path=None, device="cpu"):
    """Загружает модель классификации ритма (3 класса)."""
    from src.models.cnn_model import ResNet1D
    
    if model_path is None:
        model_path = str(Path(__file__).parent.parent.parent / "models" / "rhythm_model.pt")
    if not Path(model_path).exists():
        return None
    
    model = ResNet1D(dropout=0.3)
    model.fc = nn.Linear(256, 3)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model.to(device)


def detect_rhythm(cycles, rhythm_model, device="cpu"):
    """Определяет ритм по ЭКГ-циклам. Возвращает (название, уверенность)."""
    if rhythm_model is None:
        return "Синусовый ритм", 0.0
    with torch.no_grad():
        x = torch.tensor(cycles, dtype=torch.float32).to(device)
        logits = rhythm_model(x)
        probs = torch.softmax(logits, dim=1)
        avg = probs.mean(dim=0)
        pred = avg.argmax().item()
        conf = avg[pred].item()
    return RHYTHM_CLASSES.get(pred, "Синусовый ритм"), conf


def load_model(model_path=None, device="cpu"):
    """Загружает модель. Ищет _full.pt, затем _encoder.pt."""
    from src.models.cnn_model import ResNet1D

    if model_path is None:
        model_path = "models/resnet1d_full.pt"

    model = ResNet1D(dropout=0.3)

    full_path = str(Path(model_path).with_suffix('')) + "_full.pt"
    enc_path = str(Path(model_path).with_suffix('')) + "_encoder.pt"

    loaded = False

    # Try full model
    full_file = Path(str(model_path).replace("_encoder.pt", "_full.pt"))
    if not full_file.exists():
        full_file = Path(model_path)
    if not full_file.exists():
        full_file = Path("models/resnet1d_full.pt")

    if full_file.exists():
        try:
            ckpt = torch.load(str(full_file), map_location=device)
            model.load_state_dict(ckpt, strict=False)
            missing = [k for k in model.state_dict() if k not in ckpt]
            if len(missing) < 5:
                loaded = True
        except Exception:
            pass

    # Try encoder as fallback
    if not loaded:
        enc_file = Path(str(full_file).replace("_full.pt", "_encoder.pt"))
        if enc_file.exists():
            try:
                ckpt = torch.load(str(enc_file), map_location=device)
                # Map Sequential indices to named keys
                encoder_keys = list(model.get_encoder().state_dict().keys())
                seq_keys = list(ckpt.keys())
                mapping = {}
                for sk, ek in zip(seq_keys, encoder_keys):
                    mapping[ek] = ckpt[sk]
                model.load_state_dict(mapping, strict=False)
                loaded = True
            except Exception:
                pass

    if not loaded:
        print("  WARN: No model found, using untrained weights")

    model = model.to(device)
    model.eval()
    return model


def preprocess_ecg_for_inference(signal, fs=500):
    """Предобработка ЭКГ: фильтр, R-пики, сегментация, нормализация.
    Возвращает: [n_cycles, 12, 350] или None при ошибке.
    """
    from src.preprocessing.filters import preprocess_ecg_signal
    from src.preprocessing.segmentation import extract_heartbeats, segment_all_leads

    signal = preprocess_ecg_signal(signal, fs)
    signal[:, 3] *= -1.0  # aVR invert

    lead_energy = [np.std(signal[:, ch]) for ch in range(min(4, signal.shape[1]))]
    best_lead = np.argmax(lead_energy) if lead_energy else 1

    beats = extract_heartbeats(signal, fs, lead_idx=best_lead)
    if len(beats['r_peaks']) < 3:
        return None, None

    cycles = segment_all_leads(signal, fs, beats['r_peaks'])
    if len(cycles) == 0:
        return None, None

    for c in range(len(cycles)):
        mean_vals = np.mean(cycles[c], axis=0, keepdims=True)
        std_vals = np.std(cycles[c], axis=0, keepdims=True)
        cycles[c] = (cycles[c] - mean_vals) / np.maximum(std_vals, 1e-8)

    return np.transpose(cycles, (0, 2, 1)).astype(np.float32), beats['r_peaks']


def predict_with_uncertainty(model, cycles, n_samples=50, device="cpu", temperature=1.0):
    """MC Dropout: среднее + 95% CI. temperature > 1 = калибровка."""
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.apply(enable_dropout)
    all_probas = []

    with torch.no_grad():
        for _ in range(n_samples):
            batch = torch.tensor(cycles, dtype=torch.float32).to(device)
            logits = model(batch)
            probas = torch.sigmoid(logits / temperature)
            all_probas.append(probas.cpu().numpy())

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


def run_inference(signal, clinical, model=None, fs=500, model_path="models/resnet1d_encoder.pt", device="cpu"):
    """
    Полный пайплайн: от сигнала до результата.

    Args:
        signal: np.ndarray [samples, 12]
        clinical: dict {'age': 62, 'sex': 'M'}
        model: nn.Module или None (загрузит из model_path)
        fs: частота дискретизации (определяется автоматически при загрузке файла)
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
    cycles, r_peaks = preprocess_ecg_for_inference(signal, fs)
    if cycles is None:
        return {'error': 'Не удалось обработать ЭКГ: не найдены R-пики'}

    # Model inference with MC Dropout
    T = 0.5  # Temperature Scaling: T<1 расширяет диапазон вероятностей
    uncertainty = predict_with_uncertainty(model, cycles, n_samples=30, device=device, temperature=T)
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

    # Heart rate from real R-peaks
    hr = estimate_heart_rate(r_peaks, fs)

    # Red flags
    red_flags = check_red_flags(signal)

    # Signal quality
    signal_quality = "Хорошее" if np.std(signal) > 1e-5 else "Плохое"
    signal_quality_detail = {
        'snr_label': "Хорошее" if np.std(signal) > 0.05 else ("Среднее" if np.std(signal) > 0.01 else "Плохое"),
        'noise_50hz': "Нет" if np.std(signal[:, 0]) < 0.1 else "Есть",
        'tremor': "Нет" if np.std(signal[:, 0]) < 0.05 else "Есть",
    }

    # Auto report with age/sex context
    age_val = clinical.get('age', None)
    sex_val = clinical.get('sex', None)
    
    # Rhythm detection
    try:
        rhythm_model = load_rhythm_model(device=device)
        rhythm_name, rhythm_conf = detect_rhythm(cycles, rhythm_model, device=device)
    except Exception:
        rhythm_name = "Синусовый ритм"
        rhythm_conf = 0.0
    
    auto_report = generate_auto_report(risk_score, gradcam_map, hr, rhythm_name, red_flags,
                                        age=age_val, sex=sex_val)

    return {
        'risk_score': risk_score,
        'risk_ci': risk_ci,
        'risk_category': risk_category,
        'gradcam_map': gradcam_map,
        'segment_table': seg_table,
        'heart_rate': hr,
        'rhythm': rhythm_name,
        'rhythm_confidence': rhythm_conf,
        'red_flags': red_flags,
        'signal_quality': signal_quality,
        'signal_quality_detail': signal_quality_detail,
        'auto_report': auto_report,
        'n_cycles': len(cycles),
    }
