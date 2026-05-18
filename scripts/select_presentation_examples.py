#!/usr/bin/env python
"""Отбор качественных ЭКГ-примеров для презентации по категориям."""
import sys, os, ast, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loader import load_single_record
from src.preprocessing.filters import preprocess_ecg_signal
from src.preprocessing.segmentation import reorder_leads_to_standard

meta = pd.read_csv('data/processed/metadata_enriched.csv')
demo_dir = Path('src/app/demo_data')
demo_dir.mkdir(parents=True, exist_ok=True)
processed_dir = Path('data/processed')

def parse_scp(scp_str):
    if isinstance(scp_str, str):
        try: return ast.literal_eval(scp_str)
        except: return {}
    return {}

def select_and_save(df, category, n_needed, min_conf=70, max_per_category=5):
    """Отбор n записей с высоким SCP confidence и сохранение."""
    selected = []
    for idx, row in df.iterrows():
        if len(selected) >= max_per_category:
            break
        codes = parse_scp(row['scp_codes'])
        # Проверка confidence для основных кодов
        has_good_conf = False
        for code, conf in codes.items():
            if 'NORM' in code or 'MI' in code or 'ASMI' in code or 'IMI' in code or 'LVH' in code:
                if conf >= min_conf:
                    has_good_conf = True
        if not has_good_conf and category != 'NORM':
            continue
        selected.append(row)
    
    if not selected and category != 'NORM':
        # Fallback: взять первые n с y_меткой = 1
        selected = df.head(max_per_category).to_dict('records')
    elif category == 'NORM':
        selected = df.head(max_per_category).to_dict('records')
    
    saved = 0
    root = Path('data/raw/ptb-xl')
    csv_candidates = list(root.rglob("ptbxl_database.csv"))
    data_root = csv_candidates[0].parent if csv_candidates else root

    for row in selected:
        try:
            signal, fs, sig_name = load_single_record(row['filename_hr'], str(data_root))
            signal = reorder_leads_to_standard(signal, sig_name)
            signal = preprocess_ecg_signal(signal, fs)
            signal[:, 3] *= -1.0
            if signal.shape[0] > 5000:
                signal = signal[:5000]
            elif signal.shape[0] < 5000:
                signal = np.pad(signal, ((0, 5000 - signal.shape[0]), (0, 0)))
            
            # Проверка качества: std > 0.05, нет NaN
            if np.std(signal) < 0.05 or np.isnan(signal).any():
                continue
            
            ecg_id = row['ecg_id']
            np.save(demo_dir / f'X_{ecg_id}.npy', signal.astype(np.float32))
            saved += 1
        except Exception as e:
            print(f"  WARN: record {row.get('ecg_id', '?')} failed: {str(e)[:60]}")
    
    return saved

# Категории
categories = {
    'Normal': meta[(meta['y_acs'] == 0) & (meta['y_glzh'] == 0) & (meta['y_block'] == 0)],
    'ASMI': meta[meta['scp_codes'].str.contains('ASMI', na=False)],
    'IMI': meta[meta['scp_codes'].str.contains('IMI', na=False)],
    'ALMI': meta[meta['scp_codes'].str.contains('ALMI', na=False)],
    'AMI': meta[meta['scp_codes'].str.contains('AMI', na=False)],
    'CLBBB': meta[meta['scp_codes'].str.contains('CLBBB', na=False)],
    'CRBBB': meta[meta['scp_codes'].str.contains('CRBBB|IRBBB', na=False)],
    'LVH': meta[meta['y_glzh'] == 1],
    'AFIB': meta[meta['y_rhythm'] == 1],
    '1AVB': meta[meta['scp_codes'].str.contains('1AVB', na=False)],
}

print("Selecting and saving high-quality ECG examples...")
total = 0
for cat_name, df in categories.items():
    n = min(5, len(df))
    saved = select_and_save(df, cat_name, n)
    total += saved
    print(f"  {cat_name}: {saved}/{n} saved")

# Обновить demo_metadata.csv
demo_meta = []
for f in sorted(demo_dir.glob('X_*.npy')):
    ecg_id = int(f.stem.split('_')[1])
    row = meta[meta['ecg_id'] == ecg_id]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    codes = parse_scp(r['scp_codes'])
    # Определить диагноз
    if r['y_acs'] == 1:
        diag = 'ACS'
        if 'ASMI' in str(codes): diag = 'ASMI'
        elif 'IMI' in str(codes): diag = 'IMI'
        elif 'ALMI' in str(codes): diag = 'ALMI'
        elif 'AMI' in str(codes): diag = 'AMI'
    elif 'CLBBB' in str(codes): diag = 'CLBBB'
    elif 'CRBBB' in str(codes) or 'IRBBB' in str(codes): diag = 'CRBBB'
    elif r['y_glzh'] == 1: diag = 'LVH'
    elif r['y_rhythm'] == 1: diag = 'AFIB'
    elif '1AVB' in str(codes): diag = '1AVB'
    else: diag = 'Normal'
    
    demo_meta.append({
        'ecg_id': ecg_id, 'patient_age': r['age'], 'patient_sex': r['sex'],
        'y_acs': int(r['y_acs']), 'diagnosis': diag,
        'scp_codes': str(codes)
    })

pd.DataFrame(demo_meta).to_csv(demo_dir / 'demo_metadata.csv', index=False)
print(f"\nOK {total} new examples saved, {len(demo_meta)} total in metadata")
