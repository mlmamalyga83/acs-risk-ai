# ============================================================
# ACS ECG Detector  loading ECG from clinical devices
# ============================================================
# Supports: CSV, Philips XML, DICOM, WFDB formats.

from pathlib import Path
from typing import Tuple, Dict
import numpy as np


STANDARD_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                        'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_ecg_csv(filepath: str, fs: float = 500.0) -> Tuple[np.ndarray, float, Dict]:
    """CSV с 12 столбцами. Автоопределение заголовка и разделителя."""
    import pandas as pd
    ...


def load_ecg_philips_xml(filepath: str) -> Tuple[np.ndarray, float, Dict]:
    """Philips PageWriter XML."""
    ...


def load_ecg_dicom(filepath: str) -> Tuple[np.ndarray, float, Dict]:
    """DICOM Waveform через pydicom."""
    ...


def load_ecg_wfdb(hea_path: str) -> Tuple[np.ndarray, float, Dict]:
    """WFDB (.hea + .dat)."""
    import wfdb
    ...


def auto_load_ecg(filepath: str) -> Tuple[np.ndarray, float, Dict]:
    """
    Автоопределение формата по расширению.
    .csv/.txt  load_ecg_csv()
    .xml       load_ecg_philips_xml()
    .dcm       load_ecg_dicom()
    .hea       load_ecg_wfdb()
    """
    ext = Path(filepath).suffix.lower()
    if ext in ('.csv', '.txt'):
        return load_ecg_csv(filepath)
    elif ext == '.xml':
        return load_ecg_philips_xml(filepath)
    elif ext == '.dcm':
        return load_ecg_dicom(filepath)
    elif ext == '.hea':
        return load_ecg_wfdb(filepath)
    else:
        raise ValueError(f"Неизвестный формат: {ext}. Поддерживаются: CSV, XML (Philips), DICOM, WFDB.")


def validate_uploaded_ecg(signal: np.ndarray, fs: float) -> Tuple[bool, str]:
    """Проверка: 12 каналов, 2500 сэмплов, нет NaN, не плоская линия."""
    errors = []
    if signal.shape[1] != 12:
        errors.append(f"Ожидается 12 отведений, получено {signal.shape[1]}")
    if signal.shape[0] < 2500:
        errors.append(f"Запись слишком короткая: {signal.shape[0]} сэмплов")
    if np.isnan(signal).any():
        errors.append("Сигнал содержит NaN")
    if np.std(signal) < 1e-6:
        errors.append("Сигнал  плоская линия")
    if errors:
        return False, "\n".join(errors)
    return True, "OK"
