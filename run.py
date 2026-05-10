#!/usr/bin/env python
# ============================================================
# ACS ECG Detector — CLI-оркестратор всех этапов
# ============================================================
# Использование:
#   python run.py --stage check       Проверка системы
#   python run.py --stage init        Создание структуры
#   python run.py --stage download    Скачивание датасетов
#   python run.py --stage all         Этапы 1–7 последовательно
#   python run.py --stage all --tune  С авто-подбором гиперпараметров
#   python run.py --stage cnn --cpu-only  Ускоренное обучение на CPU
# ============================================================

import sys
import argparse
import shutil
from pathlib import Path

# Проверка Python
assert sys.version_info >= (3, 10), f"Требуется Python 3.10+, установлен: {sys.version}"


def auto_detect_device(config=None, cpu_only=False):
    """Определяет доступное устройство и адаптирует параметры."""
    import torch
    
    result = {'use_amp': False}
    
    if cpu_only:
        result['device'] = 'cpu'
        result['batch_size'] = 8
        result['epochs'] = 10
        result['architecture'] = 'simple'
        print("⚠️  CPU-only режим: 10 эпох, SimpleCNN")
        print("   AUC будет ниже на 3-5%. Для полного обучения используйте Colab GPU.")
        return result
    
    if torch.cuda.is_available():
        result['device'] = 'cuda'
        result['batch_size'] = 128
        result['use_amp'] = True
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        result['device'] = 'mps'
        result['batch_size'] = 64
        print("🖥️  Apple MPS (Metal)")
    else:
        result['device'] = 'cpu'
        result['batch_size'] = 16
        print("⚠️  GPU не обнаружен. Обучение CNN займёт >24 часов.")
        print("   Рекомендация: используйте Google Colab (бесплатный GPU T4)")
    
    return result


def preflight_check():
    """Проверка готовности системы перед запуском."""
    import psutil
    
    errors = []
    
    # Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ (установлен: {py_ver})")
    else:
        print(f"  ✅ Python {py_ver}")
    
    # Git
    if not shutil.which('git'):
        errors.append("Git не установлен (winget install Git.Git)")
    else:
        print(f"  ✅ Git {shutil.which('git')}")
    
    # Диск
    free_gb = shutil.disk_usage('.').free / 1e9
    if free_gb < 25:
        errors.append(f"Свободно {free_gb:.1f} ГБ, нужно ≥25 ГБ")
    else:
        print(f"  ✅ Диск: {free_gb:.0f} ГБ свободно")
    
    # RAM
    mem_gb = psutil.virtual_memory().total / 1e9
    if mem_gb < 8:
        errors.append(f"RAM {mem_gb:.1f} ГБ, нужно ≥8 ГБ")
    else:
        print(f"  ✅ RAM: {mem_gb:.0f} ГБ")
    
    if errors:
        print("\n❌ Ошибки:")
        for e in errors:
            print(f"   {e}")
        return False
    
    print("\n✅ Система готова")
    return True


def init_project_structure():
    """Создаёт структуру каталогов проекта."""
    dirs = [
        "data/raw/ptb-xl",
        "data/processed",
        "data/external/mit-bih-stt",
        "data/uploads",
        "config",
        "scripts",
        "src/data",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/train",
        "src/interpret",
        "src/app/demo_data",
        "src/app/utils",
        "notebooks",
        "models",
        "reports/figures",
        "reports/error_analysis",
        "runs",
        "logs",
        "docker",
        "tests",
    ]
    
    created = 0
    for d in dirs:
        p = Path(d)
        if not p.exists():
            p.mkdir(parents=True)
            (p / ".gitkeep").touch()
            created += 1
    
    print(f"✅ Создано/проверено {created} каталогов")
    
    # Проверка баз данных
    check_datasets()
    
    return True


def check_datasets():
    """Проверяет наличие и целостность баз данных."""
    import pandas as pd
    import numpy as np
    
    ptb_root = Path("data/raw/ptb-xl")
    mit_root = Path("data/external/mit-bih-stt")
    
    # PTB-XL
    csv_candidates = list(ptb_root.rglob("ptbxl_database.csv"))
    if not csv_candidates:
        print("  ⚠️  PTB-XL: ptbxl_database.csv не найден")
        print(f"     Ожидается в: {ptb_root.absolute()}")
        print("     Скопируйте содержимое D:\\ML_ECG\\PTB-XL\\ в data/raw/ptb-xl/")
    else:
        csv_path = csv_candidates[0]
        # Определить корень PTB-XL
        base = csv_path.parent
        if not (base / "RECORDS").exists():
            base = csv_path.parent.parent
        
        df = pd.read_csv(csv_path)
        dat_files = list(base.rglob("*.dat"))
        print(f"  ✅ PTB-XL: {len(df)} записей, {len(dat_files)} .dat файлов")
        if len(df) < 21000:
            print(f"     ⚠️  Ожидается ≥21,000 записей, найдено {len(df)}")
    
    # MIT-BIH
    mit_heas = list(mit_root.rglob("*.hea"))
    mit_atrs = list(mit_root.rglob("*.atr"))
    if not mit_heas:
        print("  ⚠️  MIT-BIH: .hea файлы не найдены")
        print(f"     Ожидается в: {mit_root.absolute()}")
        print("     Скопируйте содержимое D:\\ML_ECG\\mit-bih-st-change-database-1.0.0\\ в data/external/mit-bih-stt/")
    else:
        print(f"  ✅ MIT-BIH: {len(mit_heas)} записей, {len(mit_atrs)} аннотаций (.atr)")


def main():
    parser = argparse.ArgumentParser(description="ACS ECG Detector — оркестратор этапов")
    parser.add_argument("--stage", type=str, default="check",
                        choices=["check", "init", "download", "all",
                                 "eda", "preprocess", "baseline",
                                 "cnn", "multimodal", "validate", "demo", "status"])
    parser.add_argument("--tune", action="store_true", help="Авто-подбор гиперпараметров (Optuna)")
    parser.add_argument("--cpu-only", action="store_true", help="Ускоренное обучение на CPU")
    parser.add_argument("--force", action="store_true", help="Перезапустить этапы заново")
    parser.add_argument("--stop-after", type=str, default=None,
                        help="Остановиться после этапа (check/init/download/eda/preprocess/baseline)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🫀  ACS ECG Detector — v25.0")
    print("=" * 60)
    
    stage = args.stage
    
    if stage == "check":
        success = preflight_check()
        if success:
            check_datasets()
        return 0 if success else 1
    
    elif stage == "init":
        init_project_structure()
        return 0
    
    elif stage == "download":
        print("Загрузка датасетов:")
        print("  Linux/Mac: ./scripts/download_all.sh")
        print("  Windows: скачайте ZIP-архивы вручную:")
        print("    PTB-XL: https://physionet.org/content/ptb-xl/ → data/raw/ptb-xl/")
        print("    MIT-BIH: https://physionet.org/content/stdb/ → data/external/mit-bih-stt/")
        print("  После скачивания: python run.py --stage check")
        return 0
    
    elif stage in ("all", "eda", "preprocess", "baseline", "cnn", "multimodal", "validate", "demo"):
        # Проверка готовности перед запуском
        if not preflight_check():
            return 1
        
        device_info = auto_detect_device(cpu_only=args.cpu_only)
        
        # Этапы будут реализованы в соответствующих модулях
        # На Этапе 0 — только структура и проверки
        
        stages_to_run = []
        if stage == "all":
            stages_to_run = ["eda", "preprocess", "baseline", "cnn", "multimodal", "validate", "demo"]
        else:
            stages_to_run = [stage]
        
        # Фильтрация по --stop-after
        if args.stop_after and args.stop_after in stages_to_run:
            idx = stages_to_run.index(args.stop_after)
            stages_to_run = stages_to_run[:idx + 1]
        
        print(f"\nЭтапы к выполнению: {' → '.join(stages_to_run)}")
        print(f"Устройство: {device_info['device']}, batch_size: {device_info.get('batch_size')}")
        
        for s in stages_to_run:
            print(f"\n{'=' * 40}")
            print(f"Этап: {s}")
            print(f"{'=' * 40}")
            
            # Заглушки — будут заменены реальными вызовами на соответствующих этапах
            if s == "eda":
                print("  [Этап 1] EDA + разметка — будет реализован в src/data/loader.py")
            elif s == "preprocess":
                print("  [Этап 2] Предобработка — будет реализован в src/preprocessing/")
            elif s == "baseline":
                print("  [Этап 3] Бейзлайн — будет реализован в src/models/baseline.py")
            elif s == "cnn":
                print("  [Этап 4] CNN — будет реализован в src/models/cnn_model.py")
                if args.tune:
                    print("  🔧 --tune: Optuna hyperparameter sweep (20 trials)")
                if args.cpu_only:
                    print("  ⚠️  CPU-only: 10 эпох, SimpleCNN (не ResNet1D)")
            elif s == "multimodal":
                print("  [Этап 5] Multimodal — будет реализован в src/models/multimodal.py")
            elif s == "validate":
                print("  [Этап 6] Валидация — будет реализован в src/train/metrics.py")
            elif s == "demo":
                print("  [Этап 7] Streamlit — запуск src/app/main.py")
        
        print(f"\n{'=' * 60}")
        print("✅ Оркестратор завершён")
        return 0
    
    elif stage == "status":
        check_datasets()
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
