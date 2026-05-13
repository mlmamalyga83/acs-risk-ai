# ============================================================
# ACS ECG Detector  ECG signal filtering
# ============================================================

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Создаёт коэффициенты полосового фильтра Баттерворта."""
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def notch_filter(freq: float, fs: float, quality: float = 30.0):
    """Создаёт коэффициенты режекторного фильтра."""
    from scipy.signal import iirnotch
    nyq = 0.5 * fs
    return iirnotch(freq / nyq, quality)


def preprocess_ecg_signal(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    notch_freq: float = 50.0
) -> np.ndarray:
    """Полосовой 0.540 Гц + режекторный 50 Гц. filtfilt (нулевая фаза)."""
    b_band, a_band = butter_bandpass(lowcut, highcut, fs)
    b_notch, a_notch = notch_filter(notch_freq, fs)
    
    filtered = signal.copy()
    for ch in range(signal.shape[1]):
        filtered[:, ch] = filtfilt(b_band, a_band, signal[:, ch])
        filtered[:, ch] = filtfilt(b_notch, a_notch, filtered[:, ch])
    
    return filtered
