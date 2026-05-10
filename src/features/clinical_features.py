# ============================================================
# ACS ECG Detector — clinical features preprocessing
# ============================================================

import numpy as np


def preprocess_clinical(age: np.ndarray, sex: np.ndarray,
                         troponin: np.ndarray, bp_systolic: np.ndarray,
                         bp_diastolic: np.ndarray) -> np.ndarray:
    """
    Нормализует клинические признаки для модели.
    Возвращает: np.ndarray [N, 5]
    """
    age_norm = (age - 60.0) / 20.0
    troponin_log = np.log1p(troponin)
    bp_syst_norm = (bp_systolic - 135.0) / 25.0
    bp_diast_norm = (bp_diastolic - 82.0) / 15.0
    
    return np.column_stack([age_norm, sex, troponin_log, bp_syst_norm, bp_diast_norm])
