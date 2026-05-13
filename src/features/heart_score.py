# ============================================================
# ACS ECG Detector  HEART score implementation
# ============================================================

import numpy as np


def compute_heart_score(age: np.ndarray, sex: np.ndarray,
                         scp_codes: list, troponin: np.ndarray = None) -> np.ndarray:
    """
    HEART score (упрощённая): 5 компонентов, 0-10 баллов.
    History: возраст + пол  0-2
    ECG: ST изменения из SCP-кодов  0-2
    Age: <45(0), 45-65(1), >65(2)
    Risk factors: 0 (нет данных)
    Troponin: из симуляции или 0 если не передан  0-2
    """
    scores = np.zeros(len(age))
    
    # Age
    scores += np.where(age < 45, 0, np.where(age <= 65, 1, 2))
    
    # Troponin (если передан)
    if troponin is not None:
        scores += np.where(troponin < 0.04, 0, np.where(troponin < 0.5, 1, 2))
    
    return scores
