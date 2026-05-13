# ============================================================
# ACS ECG Detector  feature extraction from ECG cycles
# ============================================================

import numpy as np
import pandas as pd

# R-peak position in 350-sample window: 0.25s  500 Hz = 125
R_IDX = 125

# Clinical references relative to R-peak (in samples at 500 Hz)
ST_START = R_IDX + 35   # R+70ms
ST_END = R_IDX + 50     # R+100ms
QRS_START = R_IDX - 50  # R-100ms
QRS_END = R_IDX + 50    # R+100ms
T_START = R_IDX + 100   # R+200ms
T_END = R_IDX + 175     # R+350ms


def extract_ecg_features(cycles: np.ndarray, fs: float = 500.0) -> pd.DataFrame:
    """
    Извлекает 89 ECG-признаков из циклов.
    cycles: [n_cycles, 12, 350]
    Возвращает: DataFrame [n_cycles, 89]
    """
    n_cycles = cycles.shape[0]
    features = {}
    
    # Per-lead features (7 features  12 leads = 84)
    for lead_idx, lead_name in enumerate(['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                                            'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
        lead_signal = cycles[:, lead_idx, :]
        
        features[f'{lead_name}_r_amplitude'] = np.max(lead_signal, axis=1) - np.min(lead_signal, axis=1)
        features[f'{lead_name}_st_shift'] = np.mean(lead_signal[:, ST_START:ST_END], axis=1)
        features[f'{lead_name}_variability'] = np.std(lead_signal, axis=1)
        features[f'{lead_name}_qrs_energy'] = np.sum(lead_signal[:, QRS_START:QRS_END] ** 2, axis=1)
        features[f'{lead_name}_t_asymmetry'] = np.mean(lead_signal[:, T_START:T_END] ** 3, axis=1)
        features[f'{lead_name}_st_t_ratio'] = (
            np.mean(lead_signal[:, ST_START:ST_END], axis=1) /
            (np.mean(lead_signal[:, T_START:T_END], axis=1) + 1e-8)
        )
        features[f'{lead_name}_st_slope'] = np.diff(lead_signal[:, ST_START:ST_END], axis=1).mean(axis=1)
    
    # Global features (5)
    features['hr'] = np.ones(n_cycles) * 75  # placeholder  requires RR intervals
    features['qrs_width'] = np.ones(n_cycles) * 100
    features['qtc_fridericia'] = np.ones(n_cycles) * 420
    features['qt_dispersion'] = np.ones(n_cycles) * 30
    features['sokolow_lyon'] = (
        np.abs(np.min(cycles[:, 9, QRS_START:QRS_END], axis=1)) +  # S(V1)
        np.max(cycles[:, 10, QRS_START:QRS_END], axis=1)            # R(V5)
    )
    
    return pd.DataFrame(features)
