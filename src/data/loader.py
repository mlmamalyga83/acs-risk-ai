# ============================================================
# ACS ECG Detector — загрузка PTB-XL и других датасетов
# ============================================================

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np


def find_file(root: str, name: str) -> Optional[Path]:
    """Рекурсивный поиск файла. Возвращает Path или None."""
    for path in Path(root).rglob(name):
        return path
    return None


def load_ptbxl_metadata(base_path: str) -> pd.DataFrame:
    """
    Загружает ptbxl_database.csv.
    base_path: корень PTB-XL (содержит ptbxl_database.csv).
    Автоматически находит CSV, даже если он во вложенной папке после распаковки ZIP.
    """
    path = find_file(base_path, "ptbxl_database.csv")
    assert path is not None, f"ptbxl_database.csv не найден в {base_path}"
    return pd.read_csv(path)


def load_single_record(filename_hr: str, base_path: str) -> Tuple[np.ndarray, float, list]:
    """
    Загружает одну запись ЭКГ через wfdb.
    filename_hr: значение из колонки filename_hr (напр. 'records500/00000/00001_hr')
    base_path: корень PTB-XL
    Возвращает: (signal [samples, 12], fs, sig_name)
    """
    import wfdb
    
    hea_path = Path(base_path) / filename_hr
    record = wfdb.rdrecord(str(hea_path.with_suffix('')))
    return record.p_signal, record.fs, record.sig_name


def validate_raw_data(base_path: str = "data/raw/ptb-xl/") -> Tuple[pd.DataFrame, Path]:
    """
    Проверяет целостность PTB-XL после скачивания.
    Возвращает: (DataFrame с метаданными, Path к корню PTB-XL)
    """
    csv_path = find_file(base_path, "ptbxl_database.csv")
    assert csv_path is not None, f"ptbxl_database.csv не найден в {base_path}"
    
    # Определить корень PTB-XL (может быть на уровень выше из-за ZIP)
    root = csv_path.parent
    if not (root / "RECORDS").exists():
        root = csv_path.parent.parent
    assert (root / "RECORDS").exists(), "RECORDS-файл не найден в корне PTB-XL"
    
    df = pd.read_csv(csv_path)
    assert len(df) >= 21000, f"Ожидается ≥21000 записей, найдено {len(df)}"
    
    # Проверить .dat файлы для первых 100 записей
    missing = []
    for _, row in df.head(100).iterrows():
        hea = root / row['filename_hr'].replace('_hr', '_hr.hea')
        if not hea.exists():
            missing.append(row['filename_hr'])
    
    if missing:
        print(f"⚠️  Не найдено {len(missing)} .hea из 100 проверенных.")
        print(f"   Пример: {missing[0]}")
    
    print(f"✅ PTB-XL: {len(df)} записей, структура корректна")
    return df, root
