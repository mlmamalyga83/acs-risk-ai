#!/usr/bin/env python
# ============================================================
# ACS ECG Detector  CLI-оркестратор всех этапов
# ============================================================
# Usage:
#   python run.py --stage check       System check
#   python run.py --stage init        Create directories
#   python run.py --stage eda         Stage 1: EDA + labeling
#   python run.py --stage all         Stages 1-7
#   python run.py --stage all --tune  With hyperparameter tuning
#   python run.py --stage cnn --cpu-only  Faster CPU training
# ============================================================

import sys
import argparse
import shutil
from pathlib import Path

assert sys.version_info >= (3, 10), f"Python 3.10+ required, found: {sys.version}"


def auto_detect_device(config=None, cpu_only=False):
    import torch
    result = {'use_amp': False}
    
    if cpu_only:
        result['device'] = 'cpu'; result['batch_size'] = 8
        result['epochs'] = 10; result['architecture'] = 'simple'
        print("WARN: CPU-only mode: 10 epochs, SimpleCNN")
        print("      AUC will be 3-5% lower. Use Colab GPU for full training.")
        return result
    
    if torch.cuda.is_available():
        result['device'] = 'cuda'; result['batch_size'] = 128
        result['use_amp'] = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        result['device'] = 'mps'; result['batch_size'] = 64
        print("Apple MPS (Metal)")
    else:
        result['device'] = 'cpu'; result['batch_size'] = 16
        print("WARN: No GPU found. CNN training will take >24 hours.")
        print("      Recommendation: use Google Colab (free GPU T4)")
    return result


def preflight_check():
    import psutil
    errors = []
    
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ (found: {py_ver})")
    else:
        print(f"  OK Python {py_ver}")
    
    if not shutil.which('git'):
        errors.append("Git not installed (winget install Git.Git)")
    else:
        print(f"  OK Git {shutil.which('git')}")
    
    free_gb = shutil.disk_usage('.').free / 1e9
    if free_gb < 25:
        errors.append(f"Free space {free_gb:.1f} GB, need >=25 GB")
    else:
        print(f"  OK Disk: {free_gb:.0f} GB free")
    
    mem_gb = psutil.virtual_memory().total / 1e9
    if mem_gb < 8:
        errors.append(f"RAM {mem_gb:.1f} GB, need >=8 GB")
    else:
        print(f"  OK RAM: {mem_gb:.0f} GB")
    
    if errors:
        print("\nFAIL Errors:")
        for e in errors: print(f"   {e}")
        return False
    print("\nOK System ready")
    return True


def init_project_structure():
    dirs = [
        "data/raw/ptb-xl", "data/processed", "data/external/mit-bih-stt",
        "data/uploads", "config", "scripts",
        "src/data", "src/preprocessing", "src/features",
        "src/models", "src/train", "src/interpret",
        "src/app/demo_data", "src/app/utils",
        "notebooks", "models", "reports/figures",
        "reports/error_analysis", "runs", "logs", "docker", "tests",
    ]
    created = 0
    for d in dirs:
        p = Path(d)
        if not p.exists():
            p.mkdir(parents=True)
            (p / ".gitkeep").touch()
            created += 1
    print(f"OK {created} directories created/verified")
    check_datasets()
    return True


def check_datasets():
    import pandas as pd
    ptb_root = Path("data/raw/ptb-xl")
    mit_root = Path("data/external/mit-bih-stt")
    
    csv_candidates = list(ptb_root.rglob("ptbxl_database.csv"))
    if not csv_candidates:
        print("  WARN PTB-XL: ptbxl_database.csv not found")
        print(f"      Expected in: {ptb_root.absolute()}")
    else:
        csv_path = csv_candidates[0]
        base = csv_path.parent
        if not (base / "RECORDS").exists():
            base = csv_path.parent.parent
        df = pd.read_csv(csv_path)
        dat_files = list(base.rglob("*.dat"))
        print(f"  OK PTB-XL: {len(df)} records, {len(dat_files)} .dat files")
    
    mit_heas = list(mit_root.rglob("*.hea"))
    mit_atrs = list(mit_root.rglob("*.atr"))
    if not mit_heas:
        print("  WARN MIT-BIH: .hea files not found")
    else:
        print(f"  OK MIT-BIH: {len(mit_heas)} records, {len(mit_atrs)} annotations")


def main():
    parser = argparse.ArgumentParser(description="ACS ECG Detector - orchestrator")
    parser.add_argument("--stage", type=str, default="check",
                        choices=["check","init","download","all",
                                 "eda","preprocess","baseline",
                                 "cnn","multimodal","validate","demo","status"])
    parser.add_argument("--tune", action="store_true", help="Optuna hyperparameter tuning")
    parser.add_argument("--cpu-only", action="store_true", help="Fast CPU training")
    parser.add_argument("--force", action="store_true", help="Rerun all stages")
    parser.add_argument("--stop-after", type=str, default=None,
                        help="Stop after stage")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ACS ECG Detector - v25.0")
    print("=" * 60)
    
    stage = args.stage
    
    if stage == "check":
        success = preflight_check()
        if success: check_datasets()
        return 0 if success else 1
    
    elif stage == "init":
        init_project_structure()
        return 0
    
    elif stage == "download":
        print("Dataset download:")
        print("  Linux/Mac: ./scripts/download_all.sh")
        print("  Windows: download ZIPs manually")
        print("    PTB-XL: https://physionet.org/content/ptb-xl/")
        print("    MIT-BIH: https://physionet.org/content/stdb/")
        return 0
    
    elif stage in ("all", "eda", "preprocess", "baseline", "cnn", "multimodal", "validate", "demo"):
        if not preflight_check():
            return 1
        
        device_info = auto_detect_device(cpu_only=args.cpu_only)
        
        stages_to_run = [stage] if stage != "all" else ["eda","preprocess","baseline","cnn","multimodal","validate","demo"]
        
        if args.stop_after and args.stop_after in stages_to_run:
            idx = stages_to_run.index(args.stop_after)
            stages_to_run = stages_to_run[:idx + 1]
        
        print(f"\nStages: {' -> '.join(stages_to_run)}")
        print(f"Device: {device_info['device']}, batch={device_info.get('batch_size')}")
        
        for s in stages_to_run:
            print(f"\n{'=' * 40}")
            print(f"Stage: {s}")
            print(f"{'=' * 40}")
            
            if s == "eda":
                from src.data.loader import run_eda_stage
                run_eda_stage()
            elif s == "preprocess":
                from src.preprocessing.pipeline import run_preprocessing_stage
                run_preprocessing_stage()
            elif s == "baseline":
                from src.models.baseline import run_baseline_stage
                run_baseline_stage()
            elif s == "cnn":
                print("  [Stage 4] CNN - to be implemented")
                if args.tune:
                    print("  TUNE: Optuna hyperparameter sweep (20 trials)")
            elif s == "multimodal":
                print("  [Stage 5] Multimodal - to be implemented")
            elif s == "validate":
                print("  [Stage 6] Validation - to be implemented")
            elif s == "demo":
                print("  [Stage 7] Streamlit - launch src/app/main.py")
        
        print(f"\n{'=' * 60}")
        print("OK Orchestrator done")
        return 0
    
    elif stage == "status":
        check_datasets()
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
