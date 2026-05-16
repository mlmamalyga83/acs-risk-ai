#!/usr/bin/env python
"""Создаёт per-split multi-label файлы для обучения 4 голов (ACS, LVH, Block, Rhythm)."""
import numpy as np
import pandas as pd
from pathlib import Path

proc = Path("data/processed")
meta = pd.read_csv(proc / "metadata_enriched.csv")

# Для каждого пациента берём метки из ПЕРВОЙ записи
patient_labels = meta.groupby("patient_id")[['y_acs', 'y_glzh', 'y_block', 'y_rhythm']].first()

for split in ["train", "val", "test"]:
    pids = np.load(proc / f"patient_ids_{split}.npy")
    n_cycles = len(pids)

    # Для каждого цикла находим метки пациента
    y_multi = np.zeros((n_cycles, 4), dtype=np.int8)
    unique_pids = np.unique(pids)

    for pid in unique_pids:
        mask = pids == pid
        if pid in patient_labels.index:
            labels = patient_labels.loc[int(pid)].values.astype(np.int8)
            y_multi[mask] = labels
        else:
            print(f"  WARN: patient {pid} not in metadata, defaulting to 0")

    np.save(proc / f"y_multi_{split}.npy", y_multi)
    print(f"{split}: {n_cycles} cycles, "
          f"ACS+={y_multi[:,0].sum()}, GLZH+={y_multi[:,1].sum()}, "
          f"BLOCK+={y_multi[:,2].sum()}, RHYTHM classes={np.unique(y_multi[:,3]).tolist()}")

print("OK multi-label files created")
