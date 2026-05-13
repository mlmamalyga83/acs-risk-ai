# ============================================================
# ACS ECG Detector  загрузка PTB-XL и других датасетов
# ============================================================

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np


def find_file(root: str, name: str) -> Optional[Path]:
    """Рекурсивный поиск файла. Возвращает Path или None."""
    for path in Path(root).rglob(name):
        return path
    return None


def load_ptbxl_metadata(base_path: str) -> pd.DataFrame:
    """
    Загружает ptbxl_database.csv.
    base_path: корень PTB-XL (содержит ptbxl_database.csv).
    Автоматически находит CSV, даже если он во вложенной папке после распаковки ZIP.
    """
    path = find_file(base_path, "ptbxl_database.csv")
    assert path is not None, f"ptbxl_database.csv не найден в {base_path}"
    return pd.read_csv(path)


def load_single_record(filename_hr: str, base_path: str) -> Tuple[np.ndarray, float, list]:
    """
    Загружает одну запись ЭКГ через wfdb.
    filename_hr: значение из колонки filename_hr (напр. 'records500/00000/00001_hr')
    base_path: корень PTB-XL
    Возвращает: (signal [samples, 12], fs, sig_name)
    """
    import wfdb
    
    hea_path = Path(base_path) / filename_hr
    record = wfdb.rdrecord(str(hea_path.with_suffix('')))
    return record.p_signal, record.fs, record.sig_name


def validate_raw_data(base_path: str = "data/raw/ptb-xl/") -> Tuple[pd.DataFrame, Path]:
    """
    Проверяет целостность PTB-XL после скачивания.
    Возвращает: (DataFrame с метаданными, Path к корню PTB-XL)
    """
    csv_path = find_file(base_path, "ptbxl_database.csv")
    assert csv_path is not None, f"ptbxl_database.csv не найден в {base_path}"
    
    # Определить корень PTB-XL (может быть на уровень выше из-за ZIP)
    root = csv_path.parent
    if not (root / "RECORDS").exists():
        root = csv_path.parent.parent
    assert (root / "RECORDS").exists(), "RECORDS-файл не найден в корне PTB-XL"
    
    df = pd.read_csv(csv_path)
    assert len(df) >= 21000, f"Ожидается 21000 записей, найдено {len(df)}"
    
    # Проверить .dat файлы для первых 100 записей
    missing = []
    for _, row in df.head(100).iterrows():
        hea = root / row['filename_hr'].replace('_hr', '_hr.hea')
        if not hea.exists():
            missing.append(row['filename_hr'])
    
    if missing:
        print(f"WARN  Не найдено {len(missing)} .hea из 100 проверенных.")
        print(f"   Пример: {missing[0]}")
    
    print(f"OK PTB-XL: {len(df)} записей, структура корректна")
    return df, root


def parse_scp_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Парсит SCP-ECG коды (текстовые аббревиатуры PTB-XL) в бинарные метки.
    PTB-XL v1.0.3 использует текстовые коды: 'IMI', 'AMI', 'NORM', 'LVH', ...
    Использует ast.literal_eval() (НЕ eval!) для безопасности.
    """
    import ast
    
    # ACS (acute myocardial infarction) codes
    MI_CODES = {'IMI', 'AMI', 'ALMI', 'ASMI', 'ILMI', 'IPLMI', 'IPMI', 'LMI', 'PMI',
                'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA'}
    
    # Ischemia codes
    ISCHEMIA_CODES = {'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_',
                       'STE_', 'STD_'}
    
    # LVH / hypertrophy codes
    LVH_CODES = {'LVH', 'VCLVH', 'RVH', 'SEHYP'}
    
    # Block / conduction disorder codes
    BLOCK_CODES = {'1AVB', '2AVB', '3AVB', 'CLBBB', 'CRBBB', 'IRBBB', 'ILBBB',
                   'IVCD', 'LAFB', 'LPFB', 'WPW'}
    
    # Normal codes
    NORM_CODES = {'NORM'}
    
    # Rhythm codes -> integer class
    RHYTHM_MAP = {'SR': 0, 'AFIB': 1, 'AFLT': 2, 'STACH': 3, 'SBRAD': 4, 'PACE': 5}
    RHYTHM_ALL = set(RHYTHM_MAP.keys())
    
    def parse_one(scp_str):
        if isinstance(scp_str, str):
            codes = ast.literal_eval(scp_str)
        elif isinstance(scp_str, dict):
            codes = scp_str
        else:
            codes = {}
        
        acs = 1 if any(codes.get(c, 0) >= 70 for c in MI_CODES) else 0
        isch = 1 if any(codes.get(c, 0) >= 70 for c in ISCHEMIA_CODES) else 0
        glzh = 1 if any(codes.get(c, 0) >= 70 for c in LVH_CODES) else 0
        block = 1 if any(codes.get(c, 0) >= 70 for c in BLOCK_CODES) else 0
        norm = 1 if any(codes.get(c, 0) >= 70 for c in NORM_CODES) else 0
        
        # Rhythm: find highest-confidence rhythm code, default=0 (SR), other=6
        rhythm = 0  # default: sinus rhythm
        best_conf = 0
        for c, conf in codes.items():
            if c in RHYTHM_MAP and conf > best_conf:
                rhythm = RHYTHM_MAP[c]
                best_conf = conf
        if best_conf == 0:
            rhythm = 0 if norm == 1 else 6  # OTHER
        
        return acs, isch, glzh, block, norm, rhythm
    
    result = df['scp_codes'].apply(parse_one)
    df['y_acs'] = result.apply(lambda x: x[0])
    df['y_ischemia'] = result.apply(lambda x: x[1])
    df['y_glzh'] = result.apply(lambda x: x[2])
    df['y_block'] = result.apply(lambda x: x[3])
    df['y_norm'] = result.apply(lambda x: x[4])
    df['y_rhythm'] = result.apply(lambda x: x[5])
    
    return df


def run_eda_stage(config_path: str = "config/config.yaml"):
    """
    Этап 1: EDA + разметка данных PTB-XL.
    Вызывается из run.py --stage eda.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    from pathlib import Path
    from src.config_loader import load_config
    
    config = load_config(config_path)
    base_path = config.data.raw_path
    
    print(f"Базовый путь: {Path(base_path).absolute()}")
    
    # 1.1 Загрузка и валидация
    df, root = validate_raw_data(base_path)
    n_total = len(df)
    n_patients = df['patient_id'].nunique()
    print(f"\nЗагружено: {n_total} записей от {n_patients} пациентов")
    
    # 1.2 Разметка SCP-кодов
    df = parse_scp_codes(df)
    
    # Статистика классов
    n_acs = int(df['y_acs'].sum())
    n_isch = int(df['y_ischemia'].sum())
    n_extended = int((df['y_acs'] | df['y_ischemia']).sum())
    n_control = n_total - n_extended
    
    print(f"\n=== Распределение классов ===")
    print(f"y_primary (ОКС, 11000 conf>=70%):  {n_acs} ({n_acs/n_total*100:.1f}%)")
    print(f"y_extended (+ишемия 10300):        {n_extended} ({n_extended/n_total*100:.1f}%)")
    print(f"Контроль (без ОКС/ишемии):         {n_control} ({n_control/n_total*100:.1f}%)")
    
    # Multi-label пересечения
    n_acs_glzh = int((df['y_acs'] & df['y_glzh']).sum())
    n_acs_block = int((df['y_acs'] & df['y_block']).sum())
    n_isch_glzh = int((df['y_ischemia'] & df['y_glzh']).sum())
    n_acs_only = int((df['y_acs'] & ~df['y_glzh'] & ~df['y_block']).sum())
    
    print(f"\n=== Multi-label пересечения ===")
    print(f"ОКС + ГЛЖ одновременно:             {n_acs_glzh} ({n_acs_glzh/n_total*100:.1f}%)")
    print(f"ОКС + блокада одновременно:         {n_acs_block} ({n_acs_block/n_total*100:.1f}%)")
    print(f"Ишемия + ГЛЖ:                       {n_isch_glzh} ({n_isch_glzh/n_total*100:.1f}%)")
    print(f"Только ОКС (без ГЛЖ/блокады):       {n_acs_only} ({n_acs_only/n_total*100:.1f}%)")
    
    # Rhythm distribution
    rhythm_names = {0:'SR', 1:'AFIB', 2:'AFLT', 3:'STACH', 4:'SBRAD', 5:'PACE', 6:'OTHER'}
    print(f"\n=== Распределение ритма ===")
    for k, v in sorted(df['y_rhythm'].value_counts().items()):
        print(f"  {rhythm_names.get(k, k)}: {v} ({v/n_total*100:.1f}%)")
    
    # 1.3 Сохранение метаданных
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    output_cols = ['ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight',
                   'filename_hr', 'filename_lr',
                   'y_acs', 'y_glzh', 'y_block', 'y_rhythm',
                   'scp_codes']
    available = [c for c in output_cols if c in df.columns]
    df[available].to_csv('data/processed/metadata_enriched.csv', index=False)
    print(f"\nOK metadata_enriched.csv сохранён ({len(df)} строк)")
    
    # 1.4 Сохранение multi-label меток (4 задачи: ОКС, ГЛЖ, блокада, ритм)
    y_multi = df[['y_acs', 'y_glzh', 'y_block', 'y_rhythm']].values.astype(np.int16)
    np.save('data/processed/y_multi_all.npy', y_multi)
    print(f"OK y_multi_all.npy saved ({len(y_multi)} rows x 4 classes)")
    
    # Статистика уверенности SCP для MI кодов
    import ast
    MI_CODES_CONF = {'IMI','AMI','ALMI','ASMI','ILMI','IPLMI','IPMI','LMI','PMI',
                     'INJAL','INJAS','INJIL','INJIN','INJLA'}
    conf_mi = []
    for scp_str in df['scp_codes'].dropna():
        codes = ast.literal_eval(scp_str) if isinstance(scp_str, str) else (scp_str if isinstance(scp_str, dict) else {})
        for c in MI_CODES_CONF:
            if c in codes:
                conf_mi.append(codes[c])
    
    if conf_mi:
        print(f"\n=== SCP confidence (MI codes) ===")
        print(f"Mean confidence MI: {np.mean(conf_mi):.0f}% | Median: {np.median(conf_mi):.0f}%")
    
    # 1.5 Визуализация (первые 5 ЭКГ)
    Path('reports/figures').mkdir(parents=True, exist_ok=True)
    
    try:
        n_vis = min(5, n_total)
        vis_indices = list(df[df['y_acs'] == 1].head(3).index) + list(df[df['y_acs'] == 0].head(2).index)
        
        for i, idx in enumerate(vis_indices):
            row = df.iloc[idx]
            fname = row.get('filename_hr')
            if fname:
                try:
                    signal, fs, sig_name = load_single_record(fname, str(root))
                    from src.preprocessing.segmentation import reorder_leads_to_standard
                    signal = reorder_leads_to_standard(signal, sig_name)
                    
                    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
                    time = np.arange(signal.shape[0]) / fs
                    
                    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
                    for j, (ax, lead) in enumerate(zip(axes.flat, lead_names)):
                        if j < signal.shape[1]:
                            ax.plot(time, signal[:2500, j], 'k', linewidth=0.5)
                        ax.set_title(lead, fontsize=9)
                        ax.set_xticks([0, 1, 2, 3, 4, 5])
                        ax.grid(True, alpha=0.2)
                    
                    label = "ОКС" if row['y_acs'] == 1 else "Норма"
                    fig.suptitle(f"Запись {i+1} | Patient {row['patient_id']} | {row['age']} лет | {label}",
                                fontsize=12)
                    plt.tight_layout()
                    fig.savefig(f'reports/figures/eda_ecg_{i+1}.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  [chart] ЭКГ {i+1}: {label}  сохранена")
                except Exception as e:
                    print(f"  WARN  ЭКГ {i+1}: ошибка загрузки  {e}")
        
        # Гистограмма возраста
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[df['y_acs'] == 1]['age'].dropna(), bins=30, alpha=0.6, label=f'ОКС (n={n_acs})', color='red')
        ax.hist(df[df['y_acs'] == 0]['age'].dropna(), bins=30, alpha=0.6, label=f'Норма (n={n_total-n_acs})', color='blue')
        ax.set_xlabel('Возраст')
        ax.set_ylabel('Количество')
        ax.legend()
        ax.set_title('Распределение возраста: ОКС vs Норма')
        fig.savefig('reports/figures/eda_age_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [chart] Гистограмма возраста  сохранена")
    except Exception as e:
        print(f"  WARN  Визуализация: ошибка  {e}")
    
    print(f"\nOK Этап 1 завершён")
    print(f"Выход: data/processed/metadata_enriched.csv")
    print(f"       data/processed/y_multi_all.npy")
    print(f"       reports/figures/eda_*.png")
    
    return df, root
