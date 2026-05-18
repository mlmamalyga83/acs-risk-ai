# ============================================================
# ACS ECG Detector — генератор автоматического заключения
# ============================================================

import numpy as np

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

LEAD_GROUPS = {
    'нижней стенке': ['II', 'III', 'aVF'],
    'передне-перегородочной области': ['V1', 'V2', 'V3', 'V4'],
    'боковой стенке': ['I', 'aVL', 'V5', 'V6'],
}


def _get_affected_wall(affected_leads):
    """Определяет локализацию ИМ по поражённым отведениям."""
    for wall, leads in LEAD_GROUPS.items():
        if sum(1 for l in affected_leads if l in leads) >= 2:
            return wall
    return None


def _get_st_severity(risk_score):
    """Определяет степень выраженности ST-изменений по риску."""
    if risk_score > 0.90:
        return "выраженные"
    elif risk_score > 0.70:
        return "умеренные"
    elif risk_score > 0.50:
        return "незначительные"
    return None


def generate_auto_report(risk_score, gradcam_map, heart_rate, rhythm, red_flags,
                          age=None, sex=None):
    """Генерирует текст заключения на русском языке."""

    parts = []
    age_sex = ""
    if age and sex:
        sex_str = "мужчина" if sex == "М" else "женщина"
        age_sex = f"Пациент {sex_str} {int(age)} лет. "

    # Rhythm + HR
    if rhythm == "Синусовый":
        hr_text = f"Ритм синусовый, ЧСС {heart_rate} уд/мин."
        if heart_rate > 100:
            hr_text += " Тахикардия."
        elif heart_rate < 60:
            hr_text += " Брадикардия."
        parts.append(age_sex + hr_text)

    # Detailed risk description
    if risk_score < 0.10:
        parts.append("ЭКГ в пределах нормы. "
                     "Патологических зубцов Q, элевации или депрессии сегмента ST не выявлено. "
                     "Признаков острого коронарного синдрома нет.")

    elif risk_score < 0.25:
        parts.append("ЭКГ без явных признаков острого коронарного синдрома. "
                     "Отмечаются неспецифические изменения сегмента ST и/или зубца T. "
                     "Рекомендовано наблюдение в динамике.")

    elif risk_score < 0.50:
        parts.append("ЭКГ с неспецифическими изменениями ST-T. "
                     "Риск ОКС умеренный. Требуется сопоставление с клинической картиной "
                     "(тропонин, жалобы, анамнез).")

    elif risk_score < 0.75:
        parts.append("ЭКГ-признаки острого коронарного синдрома. "
                     f"Риск ОКС: {risk_score:.0%}.")
        st_severity = _get_st_severity(risk_score)
        if st_severity:
            affected = _find_affected_leads(gradcam_map, threshold=0.5)
            if affected:
                wall = _get_affected_wall(affected)
                if wall:
                    parts.append(f"Определяются {st_severity} ST-изменения в области {wall}. "
                                 f"Требуется экстренная консультация кардиолога.")
                else:
                    parts.append(f"Определяются {st_severity} ST-изменения в отведениях "
                                 f"{', '.join(affected[:4])}.")
            else:
                parts.append("ST-изменения не выражены, требуется дополнительная оценка.")

    else:  # risk_score >= 0.75
        parts.append("ЭКГ-картина острого коронарного синдрома. "
                     f"Риск ОКС: {risk_score:.0%}.")
        affected = _find_affected_leads(gradcam_map, threshold=0.5)
        if affected:
            wall = _get_affected_wall(affected)
            if wall:
                parts.append(f"Выраженные ST-изменения в области {wall}. "
                             f"Пациент подлежит экстренной госпитализации в ОРИТ.")
            else:
                parts.append(f"Выраженные ST-изменения в отведениях: "
                             f"{', '.join(affected[:6])}.")
        else:
            parts.append("Выраженные изменения сегмента ST и зубца T.")

    # Red flags
    if red_flags.get('tela'):
        parts.append("Дополнительно: обнаружен паттерн S1Q3T3 — возможна ТЭЛА. "
                     "Рекомендовано проверить D-димер, выполнить ЭхоКГ.")
    if red_flags.get('hyperk'):
        parts.append("Дополнительно: высокий острый зубец T — возможна гиперкалиемия. "
                     "Рекомендовано проверить уровень калия крови.")

    parts.append("")
    parts.append("⚠️ Заключение сгенерировано исследовательским прототипом. "
                 "Требуется верификация врачом.")

    return "\n".join(parts)


def _find_affected_leads(gradcam_map, threshold=0.5):
    """Находит отведения с высокой активацией Grad-CAM."""
    affected = []
    if gradcam_map is None:
        return affected
    for i, name in enumerate(LEAD_NAMES):
        if i < gradcam_map.shape[0] and np.mean(np.abs(gradcam_map[i])) > threshold:
            affected.append(name)
    return affected
