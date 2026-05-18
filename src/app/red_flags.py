# ============================================================
# ACS ECG Detector — детекция red flags на ЭКГ
# ============================================================

import numpy as np


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def check_red_flags(ecg_signal, fs=500):
    """
    Детекция red flags на ЭКГ.

    Args:
        ecg_signal: np.ndarray [samples, 12]
        fs: частота дискретизации

    Returns:
        dict: {'tela': bool, 'hyperk': bool, 'details': str}
    """
    result = {'tela': False, 'hyperk': False, 'details': []}

    if ecg_signal.shape[1] < 3:
        return result

    # S1Q3T3 (ТЭЛА): S в I, Q в III, инвертированный T в III
    lead_I = ecg_signal[:, 0]
    lead_III = ecg_signal[:, 2]

    # S в I (отрицательный зубец после R)
    neg_after_r_I = np.min(lead_I[len(lead_I)//4:3*len(lead_I)//4])
    s_in_I = neg_after_r_I < -0.1 * np.max(np.abs(lead_I))

    # Q в III (отрицательный зубец перед R)
    q_in_III = np.min(lead_III[:len(lead_III)//4]) < -0.1 * np.max(np.abs(lead_III))

    # T- в III (отрицательный T)
    last_third = lead_III[2*len(lead_III)//3:]
    t_neg_III = np.mean(last_third) < -0.05 * np.max(np.abs(lead_III))

    if s_in_I and q_in_III and t_neg_III:
        result['tela'] = True
        result['details'].append("S1Q3T3 — возможна ТЭЛА")

    # Гиперкалиемия: высокий острый T (амплитуда > 80% R, T/R > 0.8)
    for lead_idx in [4, 5, 6, 7, 8]:  # V2-V6
        if lead_idx >= ecg_signal.shape[1]:
            continue
        lead = ecg_signal[:, lead_idx]
        qrs_zone = lead[len(lead)//4:3*len(lead)//4]
        t_zone = lead[3*len(lead)//4:]
        r_amp = np.max(qrs_zone) - np.min(qrs_zone)
        t_amp = np.max(t_zone) - np.min(t_zone)
        if r_amp > 0.01 and t_amp > 0.8 * r_amp:
            # Дополнительная проверка: T должен быть узким (острым)
            t_max_idx = np.argmax(np.abs(t_zone))
            t_width = np.sum(np.abs(t_zone) > 0.3 * t_amp) if t_amp > 0 else 0
            if t_width < len(t_zone) // 3:  # острый T (ширина < 1/3 зоны T)
                result['hyperk'] = True
                result['details'].append(f"Высокий острый T в {LEAD_NAMES[lead_idx]}")
                break

    return result
