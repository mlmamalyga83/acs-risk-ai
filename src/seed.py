# ============================================================
# ACS ECG Detector — централизованный random seed
# ============================================================
# Импортировать в КАЖДОМ модуле проекта.
# Обеспечивает воспроизводимость: два запуска → идентичные метрики.

RANDOM_SEED = 42
SIMULATION_SEED = 123
SHAP_SEED = 456

import random
import numpy as np
import torch

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"🔒 Random seed: {RANDOM_SEED} (numpy, random, torch)")
