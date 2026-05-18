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
    
    with open(filepath, encoding='utf-8', errors='replace') as f:
        first_line = f.readline().strip()
    
    sep = ',' if ',' in first_line else (';' if ';' in first_line else ('\t' if '\t' in first_line else None))
    df = pd.read_csv(filepath, sep=sep, encoding='utf-8', engine='python')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 12:
        signal = df[numeric_cols[:12]].values.astype(np.float32)
    else:
        raise ValueError(f"CSV должен содержать 12 числовых столбцов (найдено {len(numeric_cols)})")
    
    # Если есть столбец времени — исключить первый
    if signal.shape[1] > 12:
        signal = signal[:, -12:]
    
    return signal, fs, {'format': 'csv', 'columns': numeric_cols[:12]}


def load_ecg_philips_xml(filepath: str) -> Tuple[np.ndarray, float, Dict]:
    """Philips PageWriter XML."""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    ns = {'': 'http://www3.medical.philips.com'}
    leads_data = []
    fs = 500
    
    for waveform in root.iter('{http://www3.medical.philips.com}Waveform'):
        lead_name = waveform.get('leadID', '')
        if lead_name in ('I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                         'V1', 'V2', 'V3', 'V4', 'V5', 'V6'):
            try:
                sample_area = waveform.find('.//{http://www3.medical.philips.com}SampleArea')
                if sample_area is not None and sample_area.text:
                    values = [float(v) for v in sample_area.text.strip().split()]
                    leads_data.append((lead_name, np.array(values, dtype=np.float32)))
            except Exception:
                pass
        if waveform.get('sampleRate'):
            try:
                fs = float(waveform.get('sampleRate'))
            except ValueError:
                pass
    
    if len(leads_data) < 12:
        raise ValueError(f"Philips XML: найдено {len(leads_data)} отведений из 12")
    
    # Упорядочить по стандартному порядку
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_map = dict(leads_data)
    min_len = min(len(sig) for sig in lead_map.values())
    signal = np.zeros((min_len, 12), dtype=np.float32)
    for i, name in enumerate(lead_order):
        if name in lead_map:
            signal[:, i] = lead_map[name][:min_len]
    
    return signal, fs, {'format': 'philips_xml'}


def load_ecg_dicom(filepath: str) -> Tuple[np.ndarray, float, Dict]:
    """DICOM Waveform через pydicom."""
    import pydicom
    
    ds = pydicom.dcmread(filepath)
    sequences = ds.get((0x5400, 0x0100))
    if sequences is None:
        # Попробовать MultiplexGroup
        sequences = ds.get((0x0040, 0xa043), [])
    
    if not sequences:
        raise ValueError("DICOM: Waveform sequence не найден")
    
    channels = []
    fs = 500
    
    for seq in sequences:
        if hasattr(seq, 'WaveformSequence'):
            for ws in seq.WaveformSequence:
                for c in ws.ChannelDefinitionSequence:
                    ch_freq = c.get((0x003a, 0x001a), pydicom.dataelem.DataElement(0, 'DS', 500)).value
                    fs = float(ch_freq)
                    samples = c.get((0x5400, 0x1010), None)
                    if samples is not None:
                        channels.append(np.frombuffer(samples.value, dtype=np.int16).astype(np.float32))
    
    if not channels:
        # Простой вариант: поиск по тегам
        for elem in ds:
            if elem.tag.group == 0x5400 and elem.tag.element == 0x1010:
                data = np.frombuffer(elem.value, dtype=np.int16).astype(np.float32)
                n_channels = 12
                n_samples = len(data) // n_channels
                if n_samples > 0:
                    signal = data[:n_samples * n_channels].reshape(-1, n_channels)
                    return signal, fs, {'format': 'dicom'}
    
    if len(channels) >= 12:
        min_len = min(len(c) for c in channels)
        signal = np.zeros((min_len, 12), dtype=np.float32)
        for i in range(min(12, len(channels))):
            signal[:, i] = channels[i][:min_len]
        return signal, fs, {'format': 'dicom'}
    
    raise ValueError(f"DICOM: получено {len(channels)} каналов из 12")


def load_ecg_wfdb(hea_path: str) -> Tuple[np.ndarray, float, Dict]:
    """WFDB (.hea + .dat)."""
    import wfdb
    from pathlib import Path
    
    hea_path = str(hea_path)
    base = hea_path
    for ext in ['.hea', '.dat', '.atr']:
        if base.endswith(ext):
            base = base[:-len(ext)]
    
    try:
        record = wfdb.rdrecord(base)
    except Exception:
        record = wfdb.rdrecord(hea_path)
    
    sig_name = getattr(record, 'sig_name', None)
    return record.p_signal.astype(np.float32), record.fs, {'sig_name': sig_name}


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
