# ============================================================
# ACS ECG Detector — R-peak detection and cycle segmentation
# ============================================================

import numpy as np
from typing import Dict, Any, Optional, List

STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def reorder_leads_to_standard(signal: np.ndarray, sig_name: list) -> np.ndarray:
    """Переупорядочивает каналы в стандартный клинический порядок."""
    new_signal = np.zeros((signal.shape[0], 12))
    for i, lead in enumerate(STANDARD_LEADS):
        if lead in sig_name:
            new_signal[:, i] = signal[:, sig_name.index(lead)]
    return new_signal


def extract_heartbeats(
    ecg_signal: np.ndarray,
    fs: float,
    lead_idx: int = 1,  # Отведение II по умолчанию
    window_before: float = 0.25,
    window_after: float = 0.45,
    target_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Детектирует R-пики, извлекает отдельные сердечные циклы.
    При отказе отведения II — fallback на V5 (idx=10), aVF (idx=5).
    """
    import neurokit2 as nk
    
    # Попытка на основном отведении
    try:
        _, info = nk.ecg_peaks(ecg_signal[:, lead_idx], sampling_rate=int(fs))
        r_peaks = info['ECG_R_Peaks']
    except Exception:
        r_peaks = np.array([])
    
    # Fallback на V5, aVF
    fallback_leads = [10, 5]  # V5, aVF
    for fb in fallback_leads:
        if len(r_peaks) >= 3 or fb == lead_idx:
            continue
        try:
            _, info = nk.ecg_peaks(ecg_signal[:, fb], sampling_rate=int(fs))
            r_peaks = info['ECG_R_Peaks']
        except Exception:
            pass
    
    # Если всё ещё нет — wfdb fallback
    if len(r_peaks) < 3:
        try:
            import wfdb.processing
            r_peaks = wfdb.processing.gqrs_detect(ecg_signal[:, 1], fs)
        except Exception:
            r_peaks = np.array([])
    
    # Фильтр граничных R-пиков
    margin = int(window_before * fs)
    end_margin = int(window_after * fs)
    valid = (r_peaks >= margin) & (r_peaks < len(ecg_signal) - end_margin)
    r_peaks = r_peaks[valid]
    
    # Извлечение циклов
    win_before_samples = int(window_before * fs)
    win_after_samples = int(window_after * fs)
    cycle_len = win_before_samples + win_after_samples
    
    cycles = []
    for r in r_peaks:
        start = r - win_before_samples
        end = r + win_after_samples
        cycle = ecg_signal[start:end, :]
        if len(cycle) < cycle_len:
            cycle = np.pad(cycle, ((0, cycle_len - len(cycle)), (0, 0)))
        cycles.append(cycle[:cycle_len])
    
    rr_intervals = np.diff(r_peaks) / fs if len(r_peaks) > 1 else np.array([])
    
    return {
        'cycles': np.array(cycles) if cycles else np.zeros((0, cycle_len, ecg_signal.shape[1])),
        'r_peaks': r_peaks,
        'rr_intervals': rr_intervals
    }


def segment_all_leads(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    cycle_length: int = 350
) -> np.ndarray:
    """Извлекает синхронизированные окна по всем 12 отведениям."""
    win_before = int(0.25 * fs)
    win_after = int(0.45 * fs)
    
    cycles = []
    for r in r_peaks:
        start = r - win_before
        end = r + win_after
        if start < 0 or end > signal.shape[0]:
            continue
        cycle = signal[start:end, :]
        if cycle.shape[0] < cycle_length:
            cycle = np.pad(cycle, ((0, cycle_length - cycle.shape[0]), (0, 0)))
        cycles.append(cycle[:cycle_length])
    
    return np.array(cycles) if cycles else np.zeros((0, cycle_length, signal.shape[1]))
