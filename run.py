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


def run_cnn_stage(config, device_info, tune=False, resume=False):
    """Stage 4: CNN training."""
    import torch
    from src.data.loader import create_dataloaders
    from src.models.cnn_model import ResNet1D, build_model_from_params
    from src.train.trainer import train_full, auto_tune_hyperparams

    processed_path = config.data.processed_path
    batch_size = device_info.get('batch_size', 64)
    device = device_info['device']
    use_amp = device_info.get('use_amp', False)
    epochs = config.training.epochs if not device_info.get('cpu_only', False) else device_info.get('epochs', 10)

    print(f"Loading data (batch_size={batch_size})...")
    train_loader = create_dataloaders(split='train', batch_size=batch_size, processed_path=processed_path)
    val_loader = create_dataloaders(split='val', batch_size=batch_size, processed_path=processed_path)

    if tune:
        print("Starting Optuna hyperparameter search (20 trials)...")
        best_params = auto_tune_hyperparams(
            train_loader, val_loader, n_trials=20,
            device=device, use_amp=use_amp,
            processed_path=processed_path
        )
        print(f"Best params: {best_params}")
        model = build_model_from_params({'architecture': 'resnet1d', 'dropout': best_params.get('dropout', 0.3)})
    else:
        print("Using default hyperparameters from config.yaml")
        model = ResNet1D(dropout=config.model_cnn.dropout)

    train_config = {
        'device': device, 'use_amp': use_amp,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'epochs': epochs,
        'patience': config.training.patience
    }

    print(f"\nTraining {config.model_cnn.architecture} for {epochs} epochs on {device}...")
    best_auc = train_full(model, train_loader, val_loader, train_config, model_name=config.model_cnn.architecture, resume=resume)

    # Save final models
    torch.save(model.state_dict(), f"models/{config.model_cnn.architecture}_full.pt")
    torch.save(model.get_encoder().state_dict(), f"models/{config.model_cnn.architecture}_encoder.pt")
    print(f"OK models/{config.model_cnn.architecture}_encoder.pt saved")

    return best_auc


def run_ablation_study(config, device_info):
    """
    Ablation study: A(ECG-only) → B(Clinical-only) → C1(Multimodal frozen) → C2(Multimodal ft).
    Сравнивает AUC по ACS через DeLong test.
    """
    import torch
    import shutil
    from src.models.cnn_model import ResNet1D
    from src.models.multimodal import MultimodalECGNet
    from src.data.loader import create_dataloaders, ECGClinicalDataset
    from src.train.trainer import train_and_evaluate, train_multimodal_full
    from src.train.metrics import delong_roc_test
    from torch.utils.data import DataLoader
    import numpy as np

    processed_path = config.data.processed_path
    device = device_info['device']
    use_amp = device_info.get('use_amp', False)
    batch_size = device_info.get('batch_size', 64)
    lr = config.training.learning_rate
    wd = config.training.weight_decay

    print("\nLoading data for ablation...")
    ecg_val = create_dataloaders(split='val', batch_size=batch_size, processed_path=processed_path)
    clin_val = DataLoader(
        ECGClinicalDataset(split='val', processed_path=processed_path),
        batch_size=batch_size
    )

    results = {}

    # A: ECG-only (ResNet1D)
    print("\n[A] ECG-only (ResNet1D)...")
    ecg_train = create_dataloaders(split='train', batch_size=batch_size, processed_path=processed_path)
    model_a = ResNet1D(dropout=config.model_cnn.dropout)
    auc_a = train_and_evaluate(model_a, ecg_train, ecg_val, lr=lr, weight_decay=wd,
                                max_epochs=20, device=device, use_amp=use_amp)
    results['ecg_only'] = {'auc': auc_a}

    # B: Clinical-only (FFN baseline: predict ACS from age+sex)
    print("\n[B] Clinical-only (FFN)...")
    class ClinicalFFN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 32), torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    clin_train = DataLoader(
        ECGClinicalDataset(split='train', processed_path=processed_path),
        batch_size=batch_size
    )

    def train_clinical_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        for batch_ecg, batch_clin, batch_y, _ in loader:
            batch_clin, batch_y = batch_clin.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(batch_clin), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate_clinical(model, loader, device):
        model.eval()
        all_p, all_l, all_pid = [], [], []
        with torch.no_grad():
            for _, batch_clin, batch_y, batch_pid in loader:
                out = torch.sigmoid(model(batch_clin.to(device))).cpu().numpy()
                all_p.extend(out)
                all_l.extend(batch_y.numpy())
                all_pid.extend(batch_pid.numpy())
        from src.train.trainer import aggregate_cycle_predictions
        return aggregate_cycle_predictions(np.array(all_p), np.array(all_pid), np.array(all_l))

    model_b = ClinicalFFN().to(device)
    crit_b = torch.nn.BCEWithLogitsLoss()
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=wd)
    best_auc_b = 0.0
    for ep in range(20):
        loss = train_clinical_epoch(model_b, clin_train, crit_b, opt_b, device)
        auc = validate_clinical(model_b, clin_val, device)
        print(f"  clinical epoch {ep+1:2d}/20 | loss: {loss:.4f} | auc: {auc:.4f}")
        best_auc_b = max(best_auc_b, auc)
    results['clinical_only'] = {'auc': best_auc_b}

    # C1: Multimodal frozen
    print("\n[C1] Multimodal frozen...")
    encoder = ResNet1D(dropout=config.model_cnn.dropout)
    model_c1 = MultimodalECGNet(encoder.get_encoder(), clinical_dim=2, embedding_dim=256)
    model_c1.freeze_encoder()
    auc_c1 = train_multimodal_full(model_c1, clin_train, clin_val,
                                    {'device': device, 'use_amp': use_amp,
                                     'learning_rate': lr, 'weight_decay': wd,
                                     'epochs': 20, 'patience': 5},
                                    model_name='multimodal_frozen')
    results['multimodal_frozen'] = {'auc': auc_c1}

    # C2: Multimodal fine-tuned
    print("\n[C2] Multimodal fine-tuned...")
    encoder = ResNet1D(dropout=config.model_cnn.dropout)
    model_c2 = MultimodalECGNet(encoder.get_encoder(), clinical_dim=2, embedding_dim=256)
    auc_c2 = train_multimodal_full(model_c2, clin_train, clin_val,
                                    {'device': device, 'use_amp': use_amp,
                                     'learning_rate': lr / 10, 'weight_decay': wd,
                                     'epochs': 20, 'patience': 5},
                                    model_name='multimodal_ft')
    results['multimodal_ft'] = {'auc': auc_c2}

    # Save best model
    best_name = max(results, key=lambda k: results[k]['auc'])
    shutil.copy(f"models/{best_name}_full.pt", "models/best_encoder.pt")
    results['best_model'] = best_name

    # Summary
    print("\n" + "=" * 60)
    print("Ablation Study Results")
    print("=" * 60)
    for name, r in results.items():
        if name != 'best_model':
            print(f"  {name:25s}: AUC = {r['auc']:.4f}")
    print(f"\n  Best model: {results['best_model']}")
    print("=" * 60)

    return results


def run_multimodal_stage(config, device_info, ablation=False, resume=False):
    """Stage 5: Multimodal training + optional ablation study."""
    import torch
    from src.models.cnn_model import ResNet1D
    from src.models.multimodal import MultimodalECGNet
    from src.data.loader import ECGClinicalDataset
    from src.train.trainer import train_multimodal_full
    from torch.utils.data import DataLoader

    processed_path = config.data.processed_path
    batch_size = device_info.get('batch_size', 64)
    device = device_info['device']
    use_amp = device_info.get('use_amp', False)
    lr = config.training.learning_rate

    if ablation:
        return run_ablation_study(config, device_info)

    print("Loading multimodal data...")
    train_dataset = ECGClinicalDataset(split='train', processed_path=processed_path)
    val_dataset = ECGClinicalDataset(split='val', processed_path=processed_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Loading pretrained ECG encoder...")
    encoder = ResNet1D(dropout=config.model_cnn.dropout)
    encoder_path = f"models/resnet1d_encoder.pt"
    if Path(encoder_path).exists():
        encoder.get_encoder().load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"  Loaded {encoder_path}")
    else:
        print(f"  WARN: {encoder_path} not found, using untrained encoder")

    model = MultimodalECGNet(encoder.get_encoder(), clinical_dim=2, embedding_dim=256)
    print(f"MultimodalECGNet: {sum(p.numel() for p in model.parameters())} params")

    train_config = {
        'device': device, 'use_amp': use_amp,
        'learning_rate': lr, 'weight_decay': config.training.weight_decay,
        'epochs': config.training.epochs, 'patience': config.training.patience
    }

    print("\nPhase 1: Training with frozen encoder...")
    model.freeze_encoder()
    auc_frozen = train_multimodal_full(model, train_loader, val_loader, train_config,
                                        model_name='multimodal_frozen', resume=resume)

    print("\nPhase 2: Fine-tuning encoder...")
    model.unfreeze_encoder()
    train_config['learning_rate'] = lr / 10
    auc_ft = train_multimodal_full(model, train_loader, val_loader, train_config,
                                    model_name='multimodal_ft', resume=resume)

    print(f"\nOK Multimodal complete. Frozen AUC: {auc_frozen:.4f}, Fine-tuned AUC: {auc_ft:.4f}")
    return max(auc_frozen, auc_ft)


def main():
    parser = argparse.ArgumentParser(description="ACS ECG Detector - orchestrator")
    parser.add_argument("--stage", type=str, default="check",
                        choices=["check","init","download","all",
                                 "eda","preprocess","baseline",
                                 "cnn","multimodal","validate","demo","status"])
    parser.add_argument("--tune", action="store_true", help="Optuna hyperparameter tuning")
    parser.add_argument("--cpu-only", action="store_true", help="Fast CPU training")
    parser.add_argument("--force", action="store_true", help="Rerun all stages")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
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
        
        from src.config_loader import load_config
        config = load_config()
        
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
                from src.models.cnn_model import ResNet1D
                from src.train.trainer import train_full, auto_tune_hyperparams
                run_cnn_stage(config, device_info, tune=args.tune, resume=args.resume)
            elif s == "multimodal":
                run_multimodal_stage(config, device_info, ablation=args.tune, resume=args.resume)
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
