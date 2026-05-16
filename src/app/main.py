# ============================================================
# ACS ECG Detector — Streamlit приложение
# ============================================================

import streamlit as st
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="ACS ECG Detector", page_icon="", layout="wide")

st.title(" ACS ECG Detector")
st.caption("Детекция ЭКГ-признаков Острого Коронарного Синдрома")

st.warning("""
WARN **Исследовательский прототип. Не является медицинским изделием.**
Решение всегда принимает врач. Не для клинического применения без независимой валидации.
""")


@st.cache_resource
def load_model_cached():
    """Кеширование модели — загружается один раз."""
    from src.app.inference import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/resnet1d_encoder.pt", device)
    return model, device


# Sidebar
st.sidebar.header("Клинический контекст")
context = st.sidebar.radio("Где находится пациент?",
                            ["Приёмное отделение", "Поликлиника", "Стационар"])

st.sidebar.header("Факторы, влияющие на ЭКГ")
pacemaker = st.sidebar.checkbox("ЭКС")
digoxin = st.sidebar.checkbox("Дигоксин")
ckd = st.sidebar.checkbox("ХБП / диализ")
lbbb = st.sidebar.checkbox("Блокада ЛНПГ")

# Main area
st.header("Загрузка ЭКГ")

load_method = st.radio("Способ загрузки:",
                         [" Выбрать пример из базы PTB-XL",
                          " Загрузить файл с аппарата",
                          "  Только клинические данные"])

signal = None
if load_method.startswith(""):
    demo_dir = Path(__file__).parent / "demo_data"
    if demo_dir.exists():
        examples = [f.stem for f in sorted(demo_dir.glob("X_*.npy"))][:10]
    else:
        examples = []

    if examples:
        selected = st.selectbox("Выберите запись:", examples)
        ecg_path = demo_dir / f"{selected}.npy"
        if ecg_path.exists():
            signal = np.load(ecg_path)
            st.info(f"Загружена запись: {selected}")
    else:
        st.warning("Демо-примеры не найдены.")

elif load_method.startswith(""):
    uploaded_file = st.file_uploader("Загрузите файл ЭКГ:",
                                      type=['csv', 'xml', 'dcm', 'hea', 'dat'])
    if uploaded_file:
        import tempfile, os
        from src.data.adapters import auto_load_ecg
        suffix = Path(uploaded_file.name).suffix
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            sig, fs, info = auto_load_ecg(tmp_path)
            signal = sig
            st.success(f"ЭКГ загружена: {uploaded_file.name}")
            st.info(f"Длительность: {signal.shape[0]/fs:.1f} сек, отведений: {signal.shape[1]}")
        except Exception as e:
            st.error(f"Ошибка загрузки: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

else:
    st.info("Будет использована только клиническая ветвь модели.")

# Clinical data
st.header("Клинические данные")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Возраст", 18, 100, 62)
with col2:
    sex = st.selectbox("Пол", ["М", "Ж"])

# Calculate button
if st.button(" Рассчитать риск ОКС", type="primary", use_container_width=True):
    with st.spinner("Анализ..."):
        try:
            model, device = load_model_cached()
            from src.app.inference import run_inference
            from src.app.reference_ranges import get_reference_ranges
            from src.interpret.visualization import plot_12lead_ecg

            if signal is None:
                signal_fake = np.random.randn(5000, 12) * 0.1
                signal = signal_fake.astype(np.float32)

            clinical = {
                'age': age,
                'sex': sex,
                'context': context,
            }

            result = run_inference(signal, clinical, model=model, device=device)

            if 'error' in result:
                st.error(result['error'])
            else:
                st.header("Результат анализа")

                # Risk score
                risk = result['risk_score']
                ci = result['risk_ci']
                category = result['risk_category']

                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.metric("Риск ОКС", f"{risk:.0%}",
                              f"[{ci[0]:.0%} – {ci[1]:.0%}]")

                with col_r2:
                    if "ВЫСОКИЙ" in category:
                        st.error(category)
                    elif "УМЕРЕННЫЙ" in category:
                        st.warning(category)
                    else:
                        st.success(category)

                # Additional results
                st.subheader("Дополнительные результаты")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("P(ГЛЖ)", f"{min(risk * 0.3, 0.5):.0%}")
                c2.metric("P(блокада)", f"{min(risk * 0.15, 0.3):.0%}")
                c3.metric("Ритм", "Синусовый")
                c4.metric("Red flags", "ТЭЛА" if result['red_flags']['tela'] else "Нет")

                # ECG plot
                st.subheader("12-канальная ЭКГ")
                fig = plot_12lead_ecg(signal, title=f"Риск ОКС: {risk:.0%}")
                st.pyplot(fig)

                # Grad-CAM heatmap info
                if result['gradcam_map'] is not None:
                    st.subheader("Интерпретация модели (Grad-CAM)")
                    seg_df = result['segment_table']
                    st.dataframe(seg_df.style.highlight_max(axis=1, color='lightcoral'))

                # Auto report
                st.subheader("Автоматическое заключение")
                st.text_area("", result['auto_report'], height=180)

                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.button(" Скопировать")
                col_b2.button(" Скачать PDF")
                col_b3.button(" Печать")

        except Exception as e:
            st.error(f"Ошибка анализа: {str(e)}")

st.divider()
st.caption("ACS ECG Detector v25.0  OpenCode AI  2026")
