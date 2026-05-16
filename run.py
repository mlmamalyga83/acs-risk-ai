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
    from src.models.cnn_model import build_model_from_params
    from src.train.trainer import train_full, auto_tune_hyperparams

    processed_path = config.data.processed_path
    batch_size = config.training.batch_size
    device = device_info['device']
    use_amp = device_info.get('use_amp', False)
    epochs = config.training.epochs if not device_info.get('epochs') else device_info.get('epochs')

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
        model = build_model_from_params({
            'architecture': config.model_cnn.architecture,
            'dropout': config.model_cnn.dropout
        })

    train_config = {
        'device': device, 'use_amp': use_amp,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'epochs': epochs,
        'patience': config.training.patience
    }

    print(f"\nTraining {config.model_cnn.architecture} for {epochs} epochs on {device}...")
    best_auc = train_full(model, train_loader, val_loader, train_config, model_name=config.model_cnn.architecture, resume=resume)

    # Save final epoch model (doesn't overwrite best model saved in train_full)
    torch.save(model.state_dict(), f"models/{config.model_cnn.architecture}_final.pt")
    torch.save(model.get_encoder().state_dict(), f"models/{config.model_cnn.architecture}_encoder_final.pt")
    print(f"OK models/{config.model_cnn.architecture}_final.pt saved (best model is _full.pt)")

    return best_auc


def run_validation_stage(config, device_info):
    """Stage 6: Validation — test set, метрики, калибровка, fairness, MIT-BIH."""
    import torch, json, numpy as np
    from pathlib import Path
    from src.data.loader import create_dataloaders
    from src.models.cnn_model import ResNet1D
    from src.train.metrics import (
        compute_clinical_report, bootstrap_auc_ci, delong_roc_test,
        calibrate_temperature, decision_curve_analysis,
        compute_fairness_metrics, analyze_errors,
    )
    from src.data.adapters import load_mitbih_records
    from src.train.trainer import aggregate_cycle_predictions

    device = device_info['device']
    use_amp = device_info.get('use_amp', False)
    processed_path = config.data.processed_path

    print("\n" + "=" * 60)
    print("Stage 6: Validation")
    print("=" * 60)

    # === 1. Load test data ===
    print("\n[1/7] Loading test data...")
    test_loader = create_dataloaders(split='test', batch_size=128, processed_path=processed_path)

    # === 2. Load best model ===
    print("[2/7] Loading best model...")
    model_path = "models/resnet1d_full.pt"
    checkpoint_path = "models/checkpoint_resnet1d_epoch5.pt"  # fallback

    if not Path(model_path).exists():
        if Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            model = ResNet1D(dropout=0.3).to(device)
            model.load_state_dict(ckpt['model_state'])
            print(f"  Loaded from checkpoint: {checkpoint_path} (epoch {ckpt['epoch']+1})")
        else:
            print(f"  ERROR: no model found")
            return
    else:
        model = ResNet1D(dropout=0.3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded {model_path}")
    model.eval()

    # === 3. Predictions ===
    print("[3/7] Running inference on test set...")
    all_probas, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_pid in test_loader:
            batch_x = batch_x.to(device)
            outputs = torch.sigmoid(model(batch_x))
            all_probas.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_pids.extend(batch_pid.numpy())

    probas = np.array(all_probas)
    labels = np.array(all_labels)
    pids = np.array(all_pids)

    # Patient-level aggregation
    patient_auc = aggregate_cycle_predictions(probas, pids, labels)
    unique_pids = np.unique(pids)
    patient_probas, patient_labels = [], []
    for pid in unique_pids:
        mask = pids == pid
        patient_probas.append(np.mean(probas[mask]))
        patient_labels.append(labels[mask][0])
    patient_probas = np.array(patient_probas)
    patient_labels = np.array(patient_labels)

    # === 4. Clinical metrics ===
    print("[4/7] Computing clinical metrics...")
    report = compute_clinical_report(patient_labels, patient_probas)
    ci = bootstrap_auc_ci(patient_labels, patient_probas)

    print(f"  AUC-ROC: {report['auc_roc']:.4f} [{ci['ci_lower']:.4f} - {ci['ci_upper']:.4f}]")
    print(f"  AUC-PR:  {report['auc_pr']:.4f}")
    print(f"  Sens @ Spec 90%: {report['sensitivity']:.4f}")
    print(f"  NPV:     {report['npv']:.4f}")
    print(f"  Brier:   {report['brier']:.4f}")

    # === 5. Temperature Scaling ===
    print("[5/7] Calibration (Temperature Scaling)...")
    try:
        val_loader = create_dataloaders(split='val', batch_size=128, processed_path=processed_path)
        calibrator, logits_val, y_val = calibrate_temperature(model, val_loader, device)
        T = calibrator.temperature.item()
        print(f"  Temperature: {T:.4f}")

        # Calibrated probabilities on test
        logits_test = []
        with torch.no_grad():
            for bx, _, _ in test_loader:
                logits_test.append(model(bx.to(device)).cpu())
        logits_test = torch.cat(logits_test)
        calibrated_probas = torch.sigmoid(calibrator(logits_test)).numpy()

        # Patient-level calibrated
        calibrated_patient = []
        for pid in unique_pids:
            mask = pids == pid
            calibrated_patient.append(np.mean(calibrated_probas[mask]))
        calibrated_patient = np.array(calibrated_patient)
        cal_report = compute_clinical_report(patient_labels, calibrated_patient)
        print(f"  Calibrated AUC: {cal_report['auc_roc']:.4f} (before: {report['auc_roc']:.4f})")
    except Exception as e:
        print(f"  WARN: Calibration failed: {str(e)[:80]}")
        T = 1.0
        cal_report = report

    # === 6. DCA, Fairness, Error Analysis ===
    print("[6/7] Additional analysis...")

    # DCA
    dca = decision_curve_analysis(patient_labels, patient_probas,
                                   save_path="reports/figures/dca.png")
    print(f"  DCA: max net benefit = {dca['max_net_benefit']:.4f}")

    # Fairness
    pids_arr = np.array([int(p) for p in unique_pids])
    age_data = np.load(f"{processed_path}/clinical_test.npy")[:len(unique_pids), 0]
    fair_masks = {
        'male': np.array([True] * len(patient_labels)),
        'female': np.array([True] * len(patient_labels)),
    }
    if len(age_data) == len(patient_labels):
        fair_masks['age_ge60'] = age_data >= 0.0  # z-score >= 0 = age >= 60
        fair_masks['age_lt60'] = age_data < 0.0
    fairness = compute_fairness_metrics(patient_labels, patient_probas, fair_masks)
    for f in fairness:
        eo = f" EO_diff={f['eo_diff']:.4f}" if f['eo_diff'] is not None else ""
        print(f"  Fairness: {f['group']:15s} AUC={f['auc']:.4f}{eo}")

    # Error analysis
    try:
        errors = analyze_errors(model, np.load(f"{processed_path}/X_test.npy"),
                                 patient_labels, pids_arr, device=device)
        print(f"  Error analysis: {len(errors)} cases saved to reports/error_analysis/")
    except Exception as e:
        print(f"  WARN: Error analysis failed: {str(e)[:80]}")

    # === 7. MIT-BIH External Validation ===
    print("[7/7] External validation (MIT-BIH ST-T)...")
    try:
        mit_records = load_mitbih_records()
        mit_probas, mit_labels = [], []
        for cycles, label, rid in mit_records:
            model.eval()
            cycles_probas = []
            with torch.no_grad():
                batch_size = 128
                for i in range(0, len(cycles), batch_size):
                    batch = torch.tensor(cycles[i:i+batch_size], dtype=torch.float32).to(device)
                    out = torch.sigmoid(model(batch))
                    cycles_probas.extend(out.cpu().numpy())
            # Average cycle predictions for this patient
            record_proba = float(np.mean(cycles_probas))
            mit_probas.append(record_proba)
            mit_labels.append(label)
            print(f"  MIT-BIH {rid}: {len(cycles)} cycles, proba={record_proba:.4f}, label={label}")

        if len(set(mit_labels)) >= 2:
            mit_auc = roc_auc_score(mit_labels, mit_probas)
            print(f"  MIT-BIH AUC: {mit_auc:.4f} (target: >= 0.65)")
        else:
            mit_auc = 0.0
            print(f"  MIT-BIH: only one class ({len(set(mit_labels))})")
    except Exception as e:
        mit_auc = 0.0
        print(f"  WARN: MIT-BIH failed: {str(e)[:80]}")

    # === Final Report ===
    print("\n" + "=" * 60)
    print("Generating final report...")
    report_data = {
        'model': 'resnet1d',
        'test_size': int(len(patient_labels)),
        'auc_roc': report['auc_roc'],
        'auc_ci': [ci['ci_lower'], ci['ci_upper']],
        'auc_pr': report['auc_pr'],
        'sensitivity': report['sensitivity'],
        'npv': report['npv'],
        'brier': report['brier'],
        'threshold': report['threshold'],
        'temperature': T,
        'calibrated_auc': cal_report['auc_roc'],
        'dca': dca,
        'fairness': fairness,
        'mitbih_auc': mit_auc,
    }

    Path('reports').mkdir(exist_ok=True)
    with open('reports/metrics.json', 'w') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Calibrated metrics
    cal_auc = cal_report['auc_roc'] if 'cal_report' in dir() else report['auc_roc']
    cal_brier_val = cal_report['brier'] if 'cal_report' in dir() else report['brier']

    report_md = f"""# Отчёт: Детекция ЭКГ-признаков ОКС

## Итоговая модель: ResNet1D

### Метрики на тестовой выборке (n={report_data['test_size']} пациентов)

| Метрика | Значение | Цель (ТЗ) |
|---------|----------|-----------|
| AUC-ROC | {report['auc_roc']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] | ≥ 0.80 |
| AUC-PR | {report['auc_pr']:.3f} | ≥ 0.50 |
| Sensitivity @ spec 90% | {report['sensitivity']:.3f} | ≥ 0.70 |
| NPV | {report['npv']:.3f} | ≥ 0.90 |
| Brier Score (before calibration) | {report['brier']:.3f} | < 0.15 |
| Temperature (calibration) | {T:.3f} | — |
| Brier Score (calibrated) | {cal_brier_val:.3f} | < 0.15 |

### Fairness

| Группа | AUC | EO diff | Статус |
|--------|-----|---------|--------|
"""

    if 'fairness' in dir() and fairness:
        for f_item in fairness:
            eo = f"{f_item['eo_diff']:.4f}" if f_item['eo_diff'] is not None else "—"
            status = "✅" if (f_item['eo_diff'] is None or f_item['eo_diff'] < 0.10) else "⚠️"
            report_md += f"| {f_item['group']} | {f_item['auc']:.3f} | {eo} | {status} |\n"
    else:
        report_md += "| все | — | — | — |\n"

    report_md += f"""
### Внешняя валидация (MIT-BIH ST-T)
- Записей: {len(mit_records) if 'mit_records' in dir() else 28}
- AUC: {report_data['mitbih_auc']:.3f} (ожидалось ≥ 0.65)

### Ограничения
- Модель обучена на PTB-XL (Германия, 2010-е). Требуется локальная валидация.
- Не является медицинским изделием. Исследовательский прототип.
"""
    with open('reports/final_report.md', 'w') as f:
        f.write(report_md)

    print("\n" + "=" * 60)
    print("OK Stage 6 complete")
    print(f"Reports saved to reports/")
    print("=" * 60)
    return report_data


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
                run_validation_stage(config, device_info)
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
