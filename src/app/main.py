# ============================================================
# ACS ECG Detector — Streamlit demo application
# ============================================================

import streamlit as st
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="ACS ECG Detector", page_icon="🫀", layout="wide")

st.title("🫀 ACS ECG Detector")
st.caption("Детекция ЭКГ-признаков Острого Коронарного Синдрома")

st.warning("""
⚠️ **Исследовательский прототип. Не является медицинским изделием.**
Решение всегда принимает врач. Не для клинического применения без независимой валидации.
""")

# Sidebar — clinical context
st.sidebar.header("Клинический контекст")
context = st.sidebar.radio("Где находится пациент?", 
                            ["Приёмное отделение", "Поликлиника", "Стационар"])

st.sidebar.header("Факторы, влияющие на ЭКГ")
pacemaker = st.sidebar.checkbox("ЭКС")
digoxin = st.sidebar.checkbox("Дигоксин")
ckd = st.sidebar.checkbox("ХБП / диализ")
lbbb = st.sidebar.checkbox("Блокада ЛНПГ")

# Main area — ECG loading
st.header("Загрузка ЭКГ")

load_method = st.radio("Способ загрузки:", 
                        ["📋 Выбрать пример из базы PTB-XL",
                         "📂 Загрузить файл с аппарата",
                         "✏️  Только клинические данные"])

if load_method.startswith("📋"):
    # Demo examples from PTB-XL
    demo_dir = Path(__file__).parent / "demo_data"
    if demo_dir.exists():
        examples = [f.stem for f in sorted(demo_dir.glob("X_*.npy"))][:10]
    else:
        examples = []
    
    if examples:
        selected = st.selectbox("Выберите запись:", examples)
        st.info(f"Выбрана запись: {selected}")
    else:
        st.warning("Демо-примеры не найдены. Выполните Этап 2 для их создания.")

elif load_method.startswith("📂"):
    uploaded_file = st.file_uploader("Загрузите файл ЭКГ:", 
                                      type=['csv', 'xml', 'dcm', 'hea', 'dat'])
    if uploaded_file:
        st.success(f"Файл загружен: {uploaded_file.name}")

else:
    st.info("Будет использована только клиническая ветвь модели (без ЭКГ).")

# Clinical data
st.header("Клинические данные")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    age = st.number_input("Возраст", 18, 100, 62)
with col2:
    sex = st.selectbox("Пол", ["М", "Ж"])
with col3:
    troponin = st.number_input("Тропонин (нг/мл)", 0.0, 100.0, 2.4, step=0.1)
with col4:
    bp_sys = st.number_input("АД сист.", 60, 250, 145)
with col5:
    bp_dia = st.number_input("АД диаст.", 30, 150, 88)

# Calculate button
if st.button("🫀 Рассчитать риск ОКС", type="primary", use_container_width=True):
    with st.spinner("Анализ..."):
        # Placeholder — real inference will use run_inference()
        st.header("Результат анализа")
        
        # Risk score
        risk = 0.78
        ci = [0.73, 0.83]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Риск ОКС", f"{risk:.0%}", f"[{ci[0]:.0%} – {ci[1]:.0%}]")
        
        # Category
        thresholds = {'Поликлиника': (0.20, 0.50), 'Приёмное отделение': (0.15, 0.40), 'Стационар': (0.10, 0.30)}
        low, high = thresholds.get(context, (0.15, 0.40))
        
        if risk <= low:
            st.success(f"✅ НИЗКИЙ РИСК — ОКС маловероятен")
        elif risk <= high:
            st.warning(f"⚡ УМЕРЕННЫЙ РИСК — рекомендуется наблюдение")
        else:
            st.error(f"⚠️ ВЫСОКИЙ РИСК — активировать протокол ОКС")
        
        # Recommendation
        st.info("📋 **Рекомендация:** Активировать протокол ОКС. Тропонин, повтор ЭКГ через 3 часа.")
        
        # Differential panel
        st.subheader("Дифференциальная панель")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("P(ОКС)", "78%")
        col2.metric("P(ГЛЖ)", "12%")
        col3.metric("P(блокада)", "5%")
        col4.metric("P(норма)", "8%")
        
        # Auto-report
        st.subheader("Автоматическое заключение")
        st.text_area("", """Синусовый ритм, ЧСС 78 уд/мин. ЭОС горизонтальная.
Элевация сегмента ST в V2-V4 до +2.1 мВ.
Патологические зубцы Q в III, aVF.
Признаки острого передне-перегородочного ИМ с вовлечением нижней стенки.

⚠️ Заключение сгенерировано исследовательским прототипом. Требуется верификация врачом.
""", height=200)
        
        col1, col2, col3 = st.columns(3)
        col1.button("📋 Скопировать")
        col2.button("📥 Скачать PDF")
        col3.button("🖨️ Печать")

st.divider()
st.caption("ACS ECG Detector v25.0 — OpenCode AI — 2026")
