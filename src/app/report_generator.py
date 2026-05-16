# ============================================================
# ACS ECG Detector — генератор автоматического заключения
# ============================================================

import numpy as np


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _find_affected_leads(gradcam_map, threshold=0.3):
    """Находит отведения с высокой активацией Grad-CAM."""
    affected = []
    for i, name in enumerate(LEAD_NAMES):
        if i < gradcam_map.shape[0] and np.mean(np.abs(gradcam_map[i])) > threshold:
            affected.append(name)
    return affected


def _find_st_segment_abnormalities(gradcam_map, r_peak_idx=125):
    """Оценка ST-изменений по Grad-CAM."""
    st_abnormal = []
    st_start, st_end = r_peak_idx + 60, r_peak_idx + 160
    for i, name in enumerate(LEAD_NAMES):
        if i < gradcam_map.shape[0]:
            st_activation = np.mean(np.abs(gradcam_map[i, st_start:st_end]))
            if st_activation > 0.4:
                st_abnormal.append((name, float(st_activation)))
    return st_abnormal


def generate_auto_report(risk_score, gradcam_map, heart_rate, rhythm, red_flags):
    """Генерирует текст заключения на русском языке."""

    parts = []

    # Rhythm
    if rhythm == "Синусовый":
        parts.append(f"Синусовый ритм, ЧСС {heart_rate} уд/мин.")

    # Risk assessment
    if risk_score < 0.15:
        parts.append("ЭКГ без признаков острого коронарного синдрома.")
    elif risk_score < 0.50:
        parts.append("ЭКГ с неспецифическими изменениями. Риск ОКС умеренный.")
    else:
        parts.append(f"ЭКГ-признаки ОКС (риск {risk_score:.0%}).")
        st_leads = _find_st_segment_abnormalities(gradcam_map)
        if st_leads:
            leads_str = ", ".join([l for l, _ in st_leads[:4]])
            parts.append(f"ST-изменения в отведениях {leads_str}.")

    # Grad-CAM affected leads
    affected = _find_affected_leads(gradcam_map)
    if affected:
        parts.append(f"Модель обращает внимание на отведения: {', '.join(affected)}.")

    # Red flags
    if red_flags.get('tela'):
        parts.append("ВНИМАНИЕ: S1Q3T3 — возможна ТЭЛА. Рекомендовано проверить D-димер.")
    if red_flags.get('hyperk'):
        parts.append("ВНИМАНИЕ: высокий острый T — возможна гиперкалиемия.")

    parts.append("")
    parts.append("⚠️ Заключение сгенерировано исследовательским прототипом. "
                 "Требуется верификация врачом.")

    return "\n".join(parts)
