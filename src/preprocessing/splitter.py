# ============================================================
# ACS ECG Detector  patient-level train/val/test split
# ============================================================

from sklearn.model_selection import GroupShuffleSplit
import numpy as np


def patient_level_split(patient_ids: np.ndarray, y: np.ndarray,
                         train_ratio: float = 0.70, val_ratio: float = 0.15,
                         random_state: int = 42):
    """Разделение 70/15/15 по patient_id. Возвращает индексы."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_ratio * 2, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(np.arange(len(y)), y, groups=patient_ids))
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_idx, y[temp_idx], groups=patient_ids[temp_idx]))
    
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]
    
    # Проверка на утечку
    assert len(set(patient_ids[train_idx]) & set(patient_ids[test_idx])) == 0, \
        "Утечка данных! Пациенты из train попали в test."
    assert len(set(patient_ids[train_idx]) & set(patient_ids[val_idx])) == 0, \
        "Утечка данных! Пациенты из train попали в val."
    
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f" Patient-level split: утечек нет")
    
    return train_idx, val_idx, test_idx
