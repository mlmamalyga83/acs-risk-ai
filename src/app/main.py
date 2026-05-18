# ============================================================
# ACS ECG Detector — Streamlit приложение
# ============================================================

import streamlit as st
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Нейросетевой анализ ЭКГ", page_icon="", layout="wide")

st.title("Нейросетевой анализ ЭКГ-признаков острого коронарного синдрома")


@st.cache_resource
def load_model_cached():
    from src.app.inference import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/resnet1d_encoder.pt", device)
    return model, device


# Sidebar
st.sidebar.header("Клинический контекст")
context = st.sidebar.radio("Где находится пациент?",
                            ["Приёмное отделение", "Поликлиника", "Стационар"])

# Main area
st.header("Загрузка ЭКГ")

load_method = st.radio("Способ загрузки:",
                         [" Выбрать пример из базы PTB-XL",
                          " Загрузить файл с аппарата"])

signal = None
signal_name = ""
fs = 500
if load_method.startswith(""):
    demo_dir = Path(__file__).parent / "demo_data"
    if demo_dir.exists():
        examples = [f.stem for f in sorted(demo_dir.glob("X_*.npy"))][:10]
    else:
        examples = []

    if examples:
        selected = st.selectbox("Выберите запись:", examples,
                                 help="Демонстрационные ЭКГ из базы PTB-XL")
        ecg_path = demo_dir / f"{selected}.npy"
        if ecg_path.exists():
            signal = np.load(ecg_path)
            signal_name = selected
            st.info(f"Загружена запись: {selected}")
    else:
        st.warning("Демо-примеры не найдены.")

else:
    uploaded_file = st.file_uploader("Загрузите файл ЭКГ:",
                                      type=['csv', 'xml', 'dcm', 'hea', 'dat'],
                                      help="Поддерживаемые форматы: CSV, Philips XML, DICOM, WFDB (.hea+.dat)")
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
            signal_name = uploaded_file.name
            st.success(f"ЭКГ загружена: {uploaded_file.name}")
            st.info(f"Длительность: {signal.shape[0]/fs:.1f} сек, отведений: {signal.shape[1]}")
        except Exception as e:
            st.error(f"Ошибка загрузки: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

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
            from src.interpret.visualization import plot_12lead_ecg
            from src.app.reference_ranges import get_reference_ranges, update_flags

            if signal is None:
                st.warning("ЭКГ не загружена. Пожалуйста, выберите запись или загрузите файл.")
                st.stop()

            clinical = {'age': age, 'sex': sex, 'context': context}
            result = run_inference(signal, clinical, fs=fs, model=model, device=device)

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
                    st.metric("Риск ОКС", f"{risk:.0%}", f"[{ci[0]:.0%} – {ci[1]:.0%}]")
                with col_r2:
                    if "ВЫСОКИЙ" in category:
                        st.error(category)
                    elif "УМЕРЕННЫЙ" in category:
                        st.warning(category)
                    else:
                        st.success(category)

                # Signal quality
                st.subheader("Качество записи")
                q = result.get('signal_quality_detail', {})
                c1, c2, c3 = st.columns(3)
                c1.metric("SNR", q.get('snr_label', '—'))
                c2.metric("Наводка 50 Гц", q.get('noise_50hz', '—'))
                c3.metric("Тремор", q.get('tremor', '—'))

                # Reference ranges
                st.subheader("Референсные значения")
                hr = result.get('heart_rate', 75)
                refs = get_reference_ranges(age=age, sex=sex)
                values = {'ЧСС': hr, 'QTc': 420, 'QRS': 98, 'PR': 160}
                refs = update_flags(refs, values)
                cols = st.columns(4)
                for i, ref in enumerate(refs):
                    val = f"{ref['value']}" if ref['value'] else "—"
                    unit = ref.get('unit', '')
                    cols[i].metric(ref['name'], f"{val} {unit}".strip(), f"{ref['ref']} {ref['flag']}".strip())

                # ECG plot - каждое отведение отдельно на всю ширину
                st.subheader("12-канальная ЭКГ")
                figs = plot_12lead_ecg(signal, fs, f"Риск ОКС: {risk:.0%}")
                for fig in figs:
                    st.pyplot(fig, use_container_width=True)

                # Conclusion
                st.subheader("Заключение")
                st.text_area("", result['auto_report'], height=180)

        except Exception as e:
            st.error(f"Ошибка анализа: {str(e)}")

st.divider()
