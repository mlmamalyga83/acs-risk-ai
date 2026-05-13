# ============================================================
# ACS ECG Detector — clinical features preprocessing
# ============================================================

import numpy as np


def preprocess_clinical(age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """
    Нормализует клинические признаки для модели.
    Только реальные данные: возраст и пол.
    Возвращает: np.ndarray [N, 2]
    """
    age_norm = (age.astype(np.float32) - 60.0) / 20.0
    return np.column_stack([age_norm, sex.astype(np.float32)])
