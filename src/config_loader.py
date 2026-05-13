# ============================================================
# ACS ECG Detector  загрузка YAML-конфигурации
# ============================================================

from omegaconf import OmegaConf
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> OmegaConf:
    """Загружает основной конфигурационный файл."""
    path = Path(config_path)
    assert path.exists(), f"Конфиг не найден: {path.absolute()}"
    return OmegaConf.load(str(path))


def load_params(params_path: str = "config/params.yaml") -> OmegaConf:
    """Загружает диапазоны гиперпараметров для grid-search."""
    path = Path(params_path)
    if not path.exists():
        return OmegaConf.create()
    return OmegaConf.load(str(path))
