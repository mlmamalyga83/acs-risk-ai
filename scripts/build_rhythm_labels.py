#!/usr/bin/env python
"""Создаёт 3-классовые метки ритма: 0=SR, 1=AFIB/AFLT, 2=OTHER/PACE."""
import numpy as np
from pathlib import Path

processed = Path("data/processed")

RHYTHM_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2}
# 0=SR → 0
# 1=AFIB, 2=AFLT → 1 (фибрилляция/трепетание)
# 3=STACH, 4=SBRAD, 5=PACE, 6=OTHER → 2 (другие)

for split in ["train", "val", "test"]:
    y_multi = np.load(processed / f"y_multi_{split}.npy")
    y_rhythm_orig = y_multi[:, 3].astype(int)
    y_rhythm3 = np.array([RHYTHM_MAP.get(v, 2) for v in y_rhythm_orig], dtype=np.int8)
    np.save(processed / f"y_rhythm3_{split}.npy", y_rhythm3)
    
    from collections import Counter
    cnt = Counter(y_rhythm3.tolist())
    print(f"{split}: SR={cnt.get(0,0)}, AFIB/AFLT={cnt.get(1,0)}, OTHER={cnt.get(2,0)}")

print("OK rhythm3 labels created")
