# ============================================================
# ACS ECG Detector — simulation of clinical data
# ============================================================

import numpy as np
from typing import Tuple, Optional

SIMULATION_SEED = 123


def generate_simulated_clinical(
    n_samples: int,
    y: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates simulated: troponin (ng/mL), bp_systolic, bp_diastolic.
    Multivariate normal with correlations (troponin↑ correlates with BP↓).
    
    For y=1 (ACS): higher troponin, lower BP
    For y=0 (normal): low troponin, normal BP
    """
    rng = np.random.RandomState(SIMULATION_SEED)
    
    if y is not None:
        # Conditional generation based on label
        troponin = np.where(
            y == 1,
            rng.lognormal(mean=3.0, sigma=1.5, size=n_samples),
            rng.lognormal(mean=0.5, sigma=1.5, size=n_samples)
        )
        bp_systolic = np.where(
            y == 1,
            rng.normal(loc=110, scale=25, size=n_samples),
            rng.normal(loc=135, scale=25, size=n_samples)
        )
        bp_diastolic = np.where(
            y == 1,
            rng.normal(loc=70, scale=15, size=n_samples),
            rng.normal(loc=82, scale=15, size=n_samples)
        )
    else:
        troponin = rng.lognormal(mean=0.5, sigma=1.5, size=n_samples)
        bp_systolic = rng.normal(loc=135, scale=25, size=n_samples)
        bp_diastolic = rng.normal(loc=82, scale=15, size=n_samples)
    
    return troponin, bp_systolic, bp_diastolic
