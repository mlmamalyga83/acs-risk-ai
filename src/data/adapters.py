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
    """Загрузка MIT-BIH ST-T (28 записей) с полной предобработкой.
    Возвращает список [(cycles, label, record_id)], cycles = [n_cycles, 12, 350]
    """
    import wfdb
    from scipy import signal as scipy_signal
    from pathlib import Path
    from src.preprocessing.filters import preprocess_ecg_signal
    from src.preprocessing.segmentation import extract_heartbeats, segment_all_leads

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
            import glob
            hea_files = glob.glob("**/300.hea", recursive=True)
            if hea_files:
                path = str(Path(hea_files[0]).parent)
            else:
                print("  WARN: MIT-BIH data not found.")
                return []

    records = []
    mit_path = Path(path)
    record_ids = list(range(300, 328))

    for rid in record_ids:
        try:
            rec = wfdb.rdrecord(str(mit_path / str(rid)))
            ann = wfdb.rdann(str(mit_path / str(rid)), 'atr')

            sig = rec.p_signal
            fs_orig = rec.fs

            # Resample to 500 Hz
            target_len = int(sig.shape[0] * 500 / fs_orig)
            n_ch = min(sig.shape[1], 12)
            sig_resampled = np.zeros((target_len, 12))
            for ch in range(n_ch):
                sig_resampled[:, ch] = scipy_signal.resample(sig[:, ch], target_len)
            for ch in range(n_ch, 12):
                sig_resampled[:, ch] = sig_resampled[:, ch % n_ch]

            # Apply standard preprocessing (filter, aVR invert)
            sig_filtered = preprocess_ecg_signal(sig_resampled, 500)
            sig_filtered[:, 3] *= -1.0

            # R-peak detection on lead with best amplitude
            lead_energy = [np.std(sig_filtered[:, ch]) for ch in range(12)]
            best_lead = np.argmax(lead_energy[:4])
            beats = extract_heartbeats(sig_filtered, 500, lead_idx=best_lead)

            if len(beats['r_peaks']) < 3:
                print(f"  WARN: MIT-BIH {rid}: too few R-peaks ({len(beats['r_peaks'])})")
                continue

            cycles = segment_all_leads(sig_filtered, 500, beats['r_peaks'])
            if len(cycles) == 0:
                continue

            # Z-score normalize per-cycle
            for c in range(len(cycles)):
                mean_vals = np.mean(cycles[c], axis=0, keepdims=True)
                std_vals = np.std(cycles[c], axis=0, keepdims=True)
                cycles[c] = (cycles[c] - mean_vals) / np.maximum(std_vals, 1e-8)

            cycles = np.transpose(cycles, (0, 2, 1))

            # Ischemia label
            has_ischemia = 0
            for sym in ann.symbol:
                if sym in ('N', 'S', 'T'):
                    has_ischemia = 1
                    break

            records.append((cycles.astype(np.float32), has_ischemia, rid))
        except Exception as e:
            print(f"  WARN: MIT-BIH record {rid} failed: {str(e)[:80]}")

    print(f"Loaded {len(records)} MIT-BIH records (with preprocessing)")
    return records
