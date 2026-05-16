# ============================================================
# ACS ECG Detector — референсные значения
# ============================================================


def get_reference_ranges(age, sex):
    """
    Возвращает референсные значения с флагами ✅/⚠️/🔴.

    Args:
        age: int
        sex: 'М' или 'Ж'

    Returns:
        list[dict]: [{'name': 'ЧСС', 'value': 78, 'ref': '60-100', 'flag': '✅'}, ...]
    """
    qtc_limits = (350, 450) if sex == 'М' else (360, 460)

    return [
        {
            'name': 'ЧСС',
            'value': None,
            'ref': '60-100',
            'unit': 'уд/мин',
            'flag': '—',
            'range': (60, 100)
        },
        {
            'name': 'QTc',
            'value': None,
            'ref': f'{qtc_limits[0]}-{qtc_limits[1]}',
            'unit': 'мс',
            'flag': '—',
            'range': qtc_limits
        },
        {
            'name': 'QRS',
            'value': None,
            'ref': '<120',
            'unit': 'мс',
            'flag': '—',
            'range': (0, 120)
        },
        {
            'name': 'PR',
            'value': None,
            'ref': '120-200',
            'unit': 'мс',
            'flag': '—',
            'range': (120, 200)
        },
    ]


def update_flags(refs, values):
    """Обновляет флаги на основе измеренных значений."""
    for ref in refs:
        if ref['name'] in values:
            val = values[ref['name']]
            ref['value'] = val
            lo, hi = ref['range']
            if lo <= val <= hi:
                ref['flag'] = '✅'
            elif val < lo * 0.8 or val > hi * 1.2:
                ref['flag'] = '🔴'
            else:
                ref['flag'] = '⚠️'
    return refs
