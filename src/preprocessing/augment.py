# ============================================================
# ACS ECG Detector — data augmentation
# ============================================================

import numpy as np


def augment_ecg_cycle(cycle: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Аугментация одного цикла ЭКГ.
    Применяется только к ОБУЧАЮЩЕЙ выборке.
    """
    rng = np.random.RandomState(seed)
    
    # Гауссов шум (1% от std)
    noise_level = 0.01
    noise = rng.normal(0, noise_level * np.std(cycle), cycle.shape)
    cycle = cycle + noise
    
    # Масштабирование [0.9, 1.1]
    scale = rng.uniform(0.9, 1.1)
    cycle = cycle * scale
    
    # Time warp [0.85, 1.15] — изменение скорости
    if rng.random() > 0.5:
        from scipy.interpolate import interp1d
        factor = rng.uniform(0.85, 1.15)
        old_indices = np.arange(cycle.shape[0])
        new_indices = np.linspace(0, cycle.shape[0] - 1, int(cycle.shape[0] * factor))
        cycle = interp1d(old_indices, cycle, axis=0, kind='linear', fill_value='extrapolate')(old_indices)
    
    return cycle
