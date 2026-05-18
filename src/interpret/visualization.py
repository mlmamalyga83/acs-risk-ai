# ============================================================
# ACS ECG Detector  visualization utilities
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def plot_12lead_ecg(signal, fs=500, title="12-канальная ЭКГ", save_path=None):
    """Визуализация 12-канальной ЭКГ — каждое отведение отдельно на всю ширину.
    Возвращает список figure (по одному на отведение) для последовательного вывода в Streamlit."""
    figs = []
    time = np.arange(signal.shape[0]) / fs
    
    for i, lead in enumerate(STANDARD_LEADS):
        fig, ax = plt.subplots(1, 1, figsize=(14, 2))
        if i < signal.shape[1]:
            ax.plot(time, signal[:, i], 'k', linewidth=0.5)
        ax.set_title(lead, fontsize=11, loc='left')
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        figs.append(fig)
    
    return figs


def plot_ecg_with_gradcam(ecg_signal, gradcam_map, title="Grad-CAM", save_path=None):
    """12-канальная ЭКГ с тепловой картой Grad-CAM."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    
    for i, (ax, lead) in enumerate(zip(axes.flat, STANDARD_LEADS)):
        if i < ecg_signal.shape[1]:
            ax.plot(ecg_signal[:, i], 'k', linewidth=0.5)
            if gradcam_map is not None and i < gradcam_map.shape[0]:
                ax.fill_between(range(len(ecg_signal[:, i])), 
                                 ecg_signal[:, i].min(), ecg_signal[:, i].max(),
                                 where=gradcam_map[i] > 0.3, alpha=0.3, color='red')
        ax.set_title(lead)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def segment_importance_table(gradcam_map, r_peak_idx=125):
    """Агрегирует Grad-CAM в таблицу P/QRS/ST/T  12 отведений."""
    segments = {
        'P': (r_peak_idx - 200, r_peak_idx - 80),
        'QRS': (r_peak_idx - 50, r_peak_idx + 50),
        'ST': (r_peak_idx + 60, r_peak_idx + 160),
        'T': (r_peak_idx + 170, r_peak_idx + 320)
    }
    
    table = {}
    for lead_idx, lead_name in enumerate(STANDARD_LEADS):
        table[lead_name] = {}
        for seg_name, (start, end) in segments.items():
            if lead_idx < gradcam_map.shape[0]:
                table[lead_name][seg_name] = np.mean(np.abs(gradcam_map[lead_idx, start:end]))
            else:
                table[lead_name][seg_name] = 0.0
    
    import pandas as pd
    return pd.DataFrame(table).T.round(3)
