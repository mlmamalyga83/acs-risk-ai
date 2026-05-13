# ============================================================
# ACS ECG Detector — preprocessing pipeline (Stage 2)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm


def run_preprocessing_stage(config_path: str = "config/config.yaml", sample_size: int = None):
    """
    Stage 2: full preprocessing pipeline.
    Memory-efficient: processes records once, saves per-split incrementally.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config_loader import load_config
    from src.data.loader import load_single_record
    from src.preprocessing.filters import preprocess_ecg_signal
    from src.preprocessing.segmentation import (
        reorder_leads_to_standard, extract_heartbeats, segment_all_leads
    )
    from src.preprocessing.augment import augment_ecg_cycle
    from src.features.clinical_features import preprocess_clinical

    config = load_config(config_path)
    
    # --- Load metadata ---
    df = pd.read_csv('data/processed/metadata_enriched.csv')
    print(f"Loaded {len(df)} records from metadata_enriched.csv")
    
    # Clean old temp directories from previous failed runs
    import shutil
    for d in ['.train', '.val', '.test']:
        p = Path(f'data/processed/{d}')
        if p.exists():
            shutil.rmtree(p)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"  TEST MODE: using first {sample_size} records only")
    
    # --- Find PTB-XL root ---
    root = Path(config.data.raw_path)
    csv_candidates = list(root.rglob("ptbxl_database.csv"))
    data_root = csv_candidates[0].parent if csv_candidates else root
    print(f"Data root: {data_root}")
    
    # --- Patient-level split assignment ---
    unique_pids = df['patient_id'].unique()
    n_patients = len(unique_pids)
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    
    n_train = int(n_patients * 0.70)
    n_val = int(n_patients * 0.15)
    
    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train:n_train + n_val])
    test_pids = set(unique_pids[n_train + n_val:])
    
    assert len(train_pids & test_pids) == 0, "Leak: train-test overlap!"
    assert len(train_pids & val_pids) == 0, "Leak: train-val overlap!"
    
    print(f"Patient split: train={len(train_pids)} val={len(val_pids)} test={len(test_pids)}")
    
    # --- Accumulators per split ---
    accumulators = {
        'train': {'X': [], 'y': [], 'pid': [], 'age': [], 'sex': []},
        'val':   {'X': [], 'y': [], 'pid': [], 'age': [], 'sex': []},
        'test':  {'X': [], 'y': [], 'pid': [], 'age': [], 'sex': []},
    }
    excluded = []
    
    def get_split(pid):
        if pid in train_pids: return 'train'
        elif pid in val_pids: return 'val'
        else: return 'test'
    
    # --- Process all records (single pass) ---
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            pid = int(row['patient_id'])
            split = get_split(pid)
            
            # Load
            signal, fs, sig_name = load_single_record(row['filename_hr'], str(data_root))
            signal = reorder_leads_to_standard(signal, sig_name)
            signal = preprocess_ecg_signal(signal, fs)
            signal[:, 3] *= -1.0  # aVR invert
            
            # R-peaks + segmentation
            beats = extract_heartbeats(signal, fs, lead_idx=1)
            if len(beats['r_peaks']) < 3:
                excluded.append({'ecg_id': row.get('ecg_id', idx), 'reason': 'no_r_peaks'})
                continue
            
            cycles = segment_all_leads(signal, fs, beats['r_peaks'])
            n_cycles = len(cycles)
            if n_cycles == 0:
                excluded.append({'ecg_id': row.get('ecg_id', idx), 'reason': 'zero_cycles'})
                continue
            
            # Normalize per-cycle
            for c in range(n_cycles):
                mean_vals = np.mean(cycles[c], axis=0, keepdims=True)
                std_vals = np.std(cycles[c], axis=0, keepdims=True)
                cycles[c] = (cycles[c] - mean_vals) / np.maximum(std_vals, 1e-8)
            
            # Transpose: [n_cycles, samples, 12] -> [n_cycles, 12, samples]
            cycles = np.transpose(cycles, (0, 2, 1))
            
            # Accumulate
            acc = accumulators[split]
            acc['X'].append(cycles)
            acc['y'].extend([int(row['y_acs'])] * n_cycles)
            acc['pid'].extend([pid] * n_cycles)
            acc['age'].extend([row['age']] * n_cycles)
            acc['sex'].extend([row['sex']] * n_cycles)
            
            # Flush if accumulator gets too large (> 30K cycles per split)
            if sum(len(arr) for arr in acc['X']) > 30_000:
                _flush_split(split, accumulators[split])
                
        except Exception as e:
            excluded.append({'ecg_id': row.get('ecg_id', idx), 'reason': str(e)[:80]})
    
    # --- Final flush for each split ---
    print("\nSaving final splits...")
    for split_name in ['train', 'val', 'test']:
        _flush_split(split_name, accumulators[split_name])
    
    # --- Simulate clinical data for each split (AFTER split) ---
    print("\nGenerating clinical data...")
    _add_clinical_and_save('train')
    _add_clinical_and_save('val')
    _add_clinical_and_save('test')
    
    # --- Save excluded ---
    pd.DataFrame(excluded).to_csv('data/processed/excluded.csv', index=False)
    n_excl = len(excluded)
    print(f"\nExcluded: {n_excl} records ({n_excl/len(df)*100:.1f}%)")
    
    # --- preprocessing_stats.json ---
    _save_preprocessing_stats()
    
    # --- Demo examples ---
    create_demo_examples(df, data_root)
    
    # --- Summary ---
    print(f"\n{'=' * 60}")
    for s in ['train', 'val', 'test']:
        if s == 'train':
            # Train saved as batches
            with open('data/processed/X_train_manifest.txt') as f:
                batches = [line.strip() for line in f]
            total = sum(len(np.load(b)) for b in batches)
            y = np.load(f'data/processed/y_{s}.npy')
            print(f"{s}: {total} cycles ({len(batches)} batch files), ACS={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
        else:
            x = np.load(f'data/processed/X_{s}.npy')
            y = np.load(f'data/processed/y_{s}.npy')
            print(f"{s}: {x.shape}, ACS={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
    print(f"{'=' * 60}")
    print("OK Stage 2 complete")


def _flush_split(split_name, acc):
    """Save accumulated cycles and clear lists."""
    if len(acc['X']) == 0:
        return
    
    X = np.concatenate(acc['X'], axis=0).astype(np.float32)
    y = np.array(acc['y'], dtype=np.int8)
    pids = np.array(acc['pid'], dtype=np.int32)
    age = np.array(acc['age'], dtype=np.float32)
    sex = np.array(acc['sex'], dtype=np.float32)
    
    save_dir = Path(f'data/processed/.{split_name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    batch_num = len(list(save_dir.glob('X_*.npy')))
    np.save(save_dir / f'X_{batch_num}.npy', X)
    np.save(save_dir / f'y_{batch_num}.npy', y)
    np.save(save_dir / f'pid_{batch_num}.npy', pids)
    np.save(save_dir / f'age_{batch_num}.npy', age)
    np.save(save_dir / f'sex_{batch_num}.npy', sex)
    
    print(f"  Flushed {split_name}: {len(X)} cycles (batch {batch_num})")
    
    # Clear accumulators
    for k in acc:
        acc[k].clear()
        if k == 'X':
            acc[k] = []
        elif k in ('y', 'pid', 'age', 'sex'):
            acc[k] = []


def _add_clinical_and_save(split_name):
    """Add real clinical data and save final split files.
    For train: augments in batches to avoid OOM."""
    import numpy as np
    from pathlib import Path
    from src.features.clinical_features import preprocess_clinical
    from src.preprocessing.augment import augment_ecg_cycle
    
    batch_dir = Path(f'data/processed/.{split_name}')
    if not batch_dir.exists():
        return
    
    # Find all batches
    batch_files = sorted(batch_dir.glob('X_*.npy'))
    batch_nums = sorted(set(int(Path(f).stem.split('_')[1]) for f in batch_files))
    
    if not batch_nums:
        return
    
    # For val/test: load all, add clinical, save
    if split_name in ('val', 'test'):
        X_parts, y_parts, pid_parts, age_parts, sex_parts = [], [], [], [], []
        for n in batch_nums:
            X_parts.append(np.load(batch_dir / f'X_{n}.npy'))
            y_parts.append(np.load(batch_dir / f'y_{n}.npy'))
            pid_parts.append(np.load(batch_dir / f'pid_{n}.npy'))
            age_parts.append(np.load(batch_dir / f'age_{n}.npy'))
            sex_parts.append(np.load(batch_dir / f'sex_{n}.npy'))
        
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts).astype(np.int8)
        pids = np.concatenate(pid_parts).astype(np.int32)
        age = np.concatenate(age_parts).astype(np.float32)
        sex = np.concatenate(sex_parts).astype(np.float32)
        
        clinical = preprocess_clinical(age, sex)
        
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        np.save(f'data/processed/X_{split_name}.npy', X)
        np.save(f'data/processed/y_{split_name}.npy', y)
        np.save(f'data/processed/clinical_{split_name}.npy', clinical.astype(np.float32))
        np.save(f'data/processed/patient_ids_{split_name}.npy', pids)
        
        import shutil
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        
        print(f"  {split_name}: {len(X)} cycles, ACS={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
        return
    
    # --- TRAIN: augment in batches to avoid OOM ---
    # First pass: calculate total size
    total_before = 0
    for n in batch_nums:
        X_part = np.load(batch_dir / f'X_{n}.npy')
        total_before += len(X_part)
    
    print(f"  train: {total_before} cycles before augmentation, augmenting in {len(batch_nums)} batches...")
    
    # Process each batch, augment, and append to output
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    X_out = None  # memory-mapped or incremental
    y_out, pid_out, clin_out = [], [], []
    
    for n in batch_nums:
        X_part = np.load(batch_dir / f'X_{n}.npy')
        y_part = np.load(batch_dir / f'y_{n}.npy').astype(np.int8)
        pid_part = np.load(batch_dir / f'pid_{n}.npy').astype(np.int32)
        age_part = np.load(batch_dir / f'age_{n}.npy').astype(np.float32)
        sex_part = np.load(batch_dir / f'sex_{n}.npy').astype(np.float32)
        
        # Augment this batch — direct allocation to avoid OOM
        n_orig = len(X_part)
        n_aug = n_orig * 2
        X_aug_batch = np.empty((n_aug, 12, 350), dtype=np.float32)
        
        for i in range(n_orig):
            X_aug_batch[i * 2] = X_part[i]
            X_aug_batch[i * 2 + 1] = augment_ecg_cycle(X_part[i])
        
        y_aug = np.repeat(y_part, 2)
        pid_aug = np.repeat(pid_part, 2)
        age_aug = np.repeat(age_part, 2)
        sex_aug = np.repeat(sex_part, 2)
        
        # Real clinical data: only age + sex, NO simulation
        clinical_aug = preprocess_clinical(age_aug, sex_aug)
        
        # Append to output arrays
        if X_out is None:
            X_out = [X_aug_batch]
        else:
            X_out.append(X_aug_batch)
        
        y_out.append(y_aug)
        pid_out.append(pid_aug)
        
        clinical_aug = preprocess_clinical(age_aug, sex_aug)
        clin_out.append(clinical_aug)
        
        print(f"    batch {n}: {len(y_part)} -> {len(y_aug)} cycles")
        del X_part, y_part, pid_part, age_part, sex_part, X_aug_batch
    
    # Write each batch as separate .npy file (proper format, no OOM)
    total_size = 0
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    for i, batch in enumerate(X_out):
        fname = f'data/processed/X_train_batch{i}.npy'
        np.save(fname, batch)
        total_size += len(batch)
        del batch
    
    # Save a manifest file listing all batches
    with open('data/processed/X_train_manifest.txt', 'w') as f:
        for i in range(len(X_out)):
            f.write(f'data/processed/X_train_batch{i}.npy\n')
    
    # Concatenate 1D arrays from per-batch lists
    y_final = np.concatenate(y_out)
    pid_final = np.concatenate(pid_out)
    clin_final = np.concatenate(clin_out, axis=0)
    
    np.save(f'data/processed/y_train.npy', y_final)
    np.save(f'data/processed/clinical_train.npy', clin_final.astype(np.float32))
    np.save(f'data/processed/patient_ids_train.npy', pid_final)
    
    import shutil
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    
    print(f"  train: {total_size} cycles saved (after augmentation), ACS={y_final.sum()}/{len(y_final)} ({y_final.mean()*100:.1f}%)")


def _save_preprocessing_stats():
    """Save normalization statistics from training data."""
    try:
        import numpy as np
        import json
        from pathlib import Path
        
        age_all = np.load('data/processed/.train/age_0.npy')
        for i in range(1, 100):
            f = Path(f'data/processed/.train/age_{i}.npy')
            if not f.exists():
                break
            age_all = np.concatenate([age_all, np.load(f)])
        
        stats = {
            'age': {'mu': float(np.mean(age_all)), 'sigma': max(float(np.std(age_all)), 1.0)},
            'troponin': {'mu_log': 1.5, 'sigma_log': 1.5},
            'bp_systolic': {'mu': 135.0, 'sigma': 25.0},
            'bp_diastolic': {'mu': 82.0, 'sigma': 15.0}
        }
    except Exception:
        stats = {
            'age': {'mu': 60.0, 'sigma': 20.0},
            'troponin': {'mu_log': 1.5, 'sigma_log': 1.5},
            'bp_systolic': {'mu': 135.0, 'sigma': 25.0},
            'bp_diastolic': {'mu': 82.0, 'sigma': 15.0}
        }
    
    Path('config').mkdir(parents=True, exist_ok=True)
    with open('config/preprocessing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


def create_demo_examples(df, data_root, n_stemi=30, n_ischemia=30, n_normal=30, n_borderline=10):
    """Creates 100 demo examples for Streamlit."""
    import numpy as np
    from pathlib import Path
    from src.data.loader import load_single_record
    from src.preprocessing.filters import preprocess_ecg_signal
    from src.preprocessing.segmentation import reorder_leads_to_standard
    
    demo_dir = Path('src/app/demo_data')
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    acs_df = df[df['y_acs'] == 1].head(n_stemi + n_borderline)
    isch_df = df[(df['y_glzh'] == 1) & (df['y_acs'] == 0)].head(n_ischemia)
    norm_df = df[(df['y_acs'] == 0) & (df['y_glzh'] == 0)].head(n_normal)
    
    demo_records = pd.concat([acs_df.head(n_stemi), isch_df, norm_df, acs_df.tail(n_borderline)])
    
    created = 0
    demo_meta = []
    
    for idx, row in demo_records.iterrows():
        try:
            signal, fs, sig_name = load_single_record(row['filename_hr'], str(data_root))
            signal = reorder_leads_to_standard(signal, sig_name)
            signal = preprocess_ecg_signal(signal, fs)
            signal[:, 3] *= -1.0
            
            if signal.shape[0] > 5000:
                signal = signal[:5000]
            elif signal.shape[0] < 5000:
                signal = np.pad(signal, ((0, 5000 - signal.shape[0]), (0, 0)))
            
            ecg_id = row.get('ecg_id', idx)
            np.save(demo_dir / f'X_{ecg_id}.npy', signal.astype(np.float32))
            
            demo_meta.append({
                'ecg_id': ecg_id, 'patient_age': row['age'],
                'patient_sex': row['sex'], 'y_acs': int(row['y_acs']),
                'diagnosis': 'ACS' if row['y_acs'] == 1 else ('LVH' if row['y_glzh'] == 1 else 'Normal')
            })
            created += 1
        except Exception:
            pass
    
    if demo_meta:
        pd.DataFrame(demo_meta).to_csv(demo_dir / 'demo_metadata.csv', index=False)
    print(f"OK {created} demo examples saved")
