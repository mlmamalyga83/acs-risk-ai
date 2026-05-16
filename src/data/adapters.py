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


def load_mitbih_records(path: str = None) -> list:
    """Загрузка MIT-BIH ST-T (28 записей). Автопоиск данных."""
    import wfdb
    from scipy import signal as scipy_signal
    from pathlib import Path

    if path is None:
        candidates = [
            "data/external/mit-bih-stt",
            "../mit-bih-st-change-database-1.0.0",
            "/root/mit-bih-st-change-database-1.0.0",
        ]
        for c in candidates:
            p = Path(c)
            if (p / "300.hea").exists() or (p / "300").with_suffix(".hea").exists():
                path = c
                break
        if path is None:
            # Find by searching
            import glob
            hea_files = glob.glob("**/300.hea", recursive=True)
            if hea_files:
                path = str(Path(hea_files[0]).parent)
            else:
                print("  WARN: MIT-BIH data not found. Suggestion:")
                print("    scp -r D:/ML_ECG/mit-bih-st-change-database-1.0.0 root@IP:data/external/mit-bih-stt/")
                return []

    records = []
    mit_path = Path(path)
    record_ids = list(range(300, 328))

    for rid in record_ids:
        try:
            rec = wfdb.rdrecord(str(mit_path / str(rid)))
            ann = wfdb.rdann(str(mit_path / str(rid)), 'atr')

            sig = rec.p_signal  # [samples, channels] 2-3 канала
            fs_orig = rec.fs  # 360 Гц

            # Ресемпл до 500 Гц
            target_len = int(sig.shape[0] * 500 / fs_orig)
            sig_resampled = np.zeros((target_len, 12))
            for ch in range(min(sig.shape[1], 12)):
                sig_resampled[:, ch] = scipy_signal.resample(sig[:, ch], target_len)

            # Паддинг до 12 каналов
            n_channels = sig.shape[1]
            if n_channels < 12:
                sig_resampled[:, n_channels:] = 0

            # Бинарная метка: есть ли ишемический эпизод
            has_ischemia = 0
            for sym in ann.symbol:
                if sym in ('N', 'S', 'T'):  # ишемические коды
                    has_ischemia = 1
                    break

            records.append((sig_resampled.astype(np.float32), has_ischemia, rid))
        except Exception as e:
            print(f"  WARN: MIT-BIH record {rid} load failed: {str(e)[:60]}")

    print(f"Loaded {len(records)} MIT-BIH records")
    return records
