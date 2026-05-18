# ============================================================
# ACS ECG Detector  training loop
# ============================================================

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from src.data.loader import create_dataloaders


class FocalLoss(nn.Module):
    """Focal Loss: фокус на сложных примерах, γ=2, α=0.25."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Обучает одну эпоху. Возвращает средний loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    report_every = max(1, n_batches // 4)
    
    for batch_idx, (batch_x, batch_y, _) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
        
        # MixUp augmentation (50% вероятности)
        if np.random.random() < 0.5:
            lam = np.random.beta(0.5, 0.5)
            perm = torch.randperm(batch_x.size(0), device=device)
            batch_x = lam * batch_x + (1 - lam) * batch_x[perm]
            batch_y = lam * batch_y + (1 - lam) * batch_y[perm]
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % report_every == 0:
            pct = (batch_idx + 1) / n_batches * 100
            print(f"  [{pct:.0f}%] {batch_idx+1}/{n_batches}, loss={loss.item():.4f}")
    
    return total_loss / len(loader)


def validate(model, loader, device):
    """Валидация с агрегацией циклпациент. Возвращает AUC."""
    model.eval()
    all_probas, all_labels, all_pids = [], [], []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_pid in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_probas.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_pids.extend(batch_pid.numpy())
    
    return aggregate_cycle_predictions(np.array(all_probas), np.array(all_pids), np.array(all_labels))


def aggregate_cycle_predictions(probas, pids, labels, method='mean'):
    """Агрегирует цикловые предсказания  пациентские."""
    unique_pids = np.unique(pids)
    patient_probas, patient_labels = [], []
    
    for pid in unique_pids:
        mask = pids == pid
        patient_probas.append(np.mean(probas[mask]))
        patient_labels.append(labels[mask][0])
    
    return roc_auc_score(np.array(patient_labels), np.array(patient_probas))


def train_full(model, train_loader, val_loader, config, model_name='model', resume=False):
    """Полный цикл обучения с Early Stopping и чекпоинтами.
    
    resume=True: автоматически найти последний чекпоинт и продолжить.
    """
    device = torch.device(config.get('device', 'cpu'))
    model = model.to(device)
    
    start_epoch = 0
    best_auc = 0.0
    patience_counter = 0
    
    # Resume from checkpoint
    if resume:
        ckpt_files = sorted(Path('models/').glob(f'checkpoint_{model_name}_epoch*.pt'))
        if ckpt_files:
            latest = ckpt_files[-1]
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            start_epoch = ckpt['epoch'] + 1
            best_auc = ckpt.get('best_auc', 0.0)
            patience_counter = ckpt.get('patience', 0)
            print(f"Resumed from {latest.name} (epoch {ckpt['epoch']+1})")
    
    pos_weight = (len(train_loader.dataset) - np.sum(train_loader.dataset.labels)) / max(np.sum(train_loader.dataset.labels), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    # Раздельные LR: encoder медленно (lr/10), FC быстро (lr*10) + защита от weight_decay
    encoder_params, fc_params = [], []
    for name, param in model.named_parameters():
        (fc_params if 'fc' in name else encoder_params).append(param)
    
    base_lr = config.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': base_lr / 10, 
         'weight_decay': config.get('weight_decay', 1e-4)},
        {'params': fc_params, 'lr': base_lr * 10, 
         'weight_decay': 1e-6},
    ])
    
    total_epochs = config.get('epochs', 50)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get('learning_rate', 0.001),
        epochs=total_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.3, div_factor=10, final_div_factor=100
    )
    
    # Fast forward scheduler to start_epoch
    if start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
    
    scaler = torch.amp.GradScaler() if config.get('use_amp', False) and device.type == 'cuda' else None
    writer = SummaryWriter(log_dir=f"runs/{model_name}")
    
    for epoch in range(start_epoch, total_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()
        
        val_auc = validate(model, val_loader, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Metrics/AUC', val_auc, epoch)
        
        print(f"Эпоха {epoch+1:2d}/{config.get('epochs', 50)} | "
              f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save encoder unconditionally after each epoch (for Colab free tier)
        torch.save(model.get_encoder().state_dict(), f"models/{model_name}_encoder.pt")
        
        # Save on improvement (full model)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"models/{model_name}_full.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Checkpoint for resume
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'best_auc': best_auc, 'patience': patience_counter
            }, f"models/checkpoint_{model_name}_epoch{epoch+1}.pt")
        
        # Early stopping
        if patience_counter >= config.get('patience', 10):
            print(f"Early Stopping на эпохе {epoch+1}")
            break
    
    # Cleanup old checkpoints (keep last 2)
    checkpoints = sorted(Path('models/').glob(f'checkpoint_{model_name}_epoch*.pt'))
    for ckpt in checkpoints[:-2]:
        ckpt.unlink()
    
    writer.close()
    
    print("=" * 60)
    print(f" ОБУЧЕНИЕ ЗАВЕРШЕНО! AUC: {best_auc:.4f}")
    print("=" * 60)
    return best_auc


def train_and_evaluate(
    model, train_loader, val_loader,
    lr=0.001, weight_decay=1e-4, max_epochs=20,
    device='cpu', use_amp=False, pos_weight=None
) -> float:
    """Быстрое обучение для Optuna/ablation. Возвращает val_auc."""
    model = model.to(device)
    if pos_weight is None:
        labels = getattr(train_loader.dataset, 'labels', None)
        if labels is not None:
            pos_weight_val = (len(labels) - np.sum(labels)) / max(np.sum(labels), 1)
        else:
            pos_weight_val = 1.0
    else:
        pos_weight_val = pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=max_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3,
        div_factor=10, final_div_factor=100
    )
    scaler = torch.amp.GradScaler() if use_amp and device == 'cuda' else None

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()
        val_auc = validate(model, val_loader, device)
        print(f"  trial epoch {epoch+1:2d}/{max_epochs} | loss: {train_loss:.4f} | auc: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  early stop at epoch {epoch+1}")
                break

    return best_auc


def auto_tune_hyperparams(train_loader, val_loader, n_trials=20, device='cpu', use_amp=False, processed_path='data/processed/'):
    """Optuna grid search: lr, dropout, batch_size, weight_decay. Возвращает best_params."""
    import optuna
    from src.models.cnn_model import ResNet1D

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        model = ResNet1D(dropout=dropout)
        loader = create_dataloaders(split='train', batch_size=batch_size, processed_path=processed_path)
        val_loader_local = create_dataloaders(split='val', batch_size=batch_size, processed_path=processed_path)

        auc = train_and_evaluate(
            model, loader, val_loader_local,
            lr=lr, weight_decay=wd, max_epochs=20,
            device=device, use_amp=use_amp
        )
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Лучшие параметры: {study.best_params} (AUC: {study.best_value:.3f})")

    from omegaconf import OmegaConf
    OmegaConf.save(OmegaConf.create(study.best_params), 'config/params.yaml')
    print("Best params saved to config/params.yaml")

    return study.best_params


def train_multimodal_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Обучает мультимодальную модель одну эпоху. Возвращает средний loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    report_every = max(1, n_batches // 4)

    for batch_idx, (batch_ecg, batch_clin, batch_y, _) in enumerate(loader):
        batch_ecg, batch_clin = batch_ecg.to(device), batch_clin.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(batch_ecg, batch_clin)
                loss = _compute_multimodal_loss(outputs, batch_y, criterion, device)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_ecg, batch_clin)
            loss = _compute_multimodal_loss(outputs, batch_y, criterion, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % report_every == 0:
            pct = (batch_idx + 1) / n_batches * 100
            print(f"  [{pct:.0f}%] {batch_idx+1}/{n_batches}, loss={loss.item():.4f}")

    return total_loss / len(loader)


def _compute_multimodal_loss(outputs, batch_y, criterion, device):
    if batch_y.dim() > 1 and batch_y.shape[1] == 4:
        batch_y = batch_y.to(device)
        loss_acs = criterion(outputs['acs'], batch_y[:, 0].float())
        loss_glzh = nn.BCEWithLogitsLoss()(outputs['glzh'], batch_y[:, 1].float())
        loss_block = nn.BCEWithLogitsLoss()(outputs['block'], batch_y[:, 2].float())
        loss_rhythm = nn.CrossEntropyLoss()(outputs['rhythm'], batch_y[:, 3].long())
        return loss_acs + 0.01 * loss_glzh + 0.01 * loss_block + 0.01 * loss_rhythm
    else:
        batch_y = batch_y.to(device).float()
        return criterion(outputs['acs'], batch_y)


def validate_multimodal(model, loader, device):
    """Валидация мультимодальной модели. Возвращает AUC по ACS."""
    model.eval()
    all_acs, all_glzh, all_block, all_rhythm = [], [], [], []
    all_labels_acs, all_labels_glzh, all_labels_block, all_labels_rhythm = [], [], [], []
    all_pids = []

    with torch.no_grad():
        for batch_ecg, batch_clin, batch_y, batch_pid in loader:
            batch_ecg = batch_ecg.to(device)
            batch_clin = batch_clin.to(device)
            outputs = model(batch_ecg, batch_clin)

            all_acs.extend(torch.sigmoid(outputs['acs']).cpu().numpy())
            all_pids.extend(batch_pid.numpy())

            if batch_y.dim() > 1 and batch_y.shape[1] == 4:
                all_labels_acs.extend(batch_y[:, 0].numpy())
                all_glzh.extend(torch.sigmoid(outputs['glzh']).cpu().numpy())
                all_labels_glzh.extend(batch_y[:, 1].numpy())
                all_block.extend(torch.sigmoid(outputs['block']).cpu().numpy())
                all_labels_block.extend(batch_y[:, 2].numpy())
                all_rhythm.extend(torch.softmax(outputs['rhythm'], dim=1).cpu().numpy())
                all_labels_rhythm.extend(batch_y[:, 3].numpy())
            else:
                all_labels_acs.extend(batch_y.numpy())

    result = {'acs_auc': aggregate_cycle_predictions(
        np.array(all_acs), np.array(all_pids), np.array(all_labels_acs))}

    if all_labels_glzh:
        result['glzh_auc'] = aggregate_cycle_predictions(
            np.array(all_glzh), np.array(all_pids), np.array(all_labels_glzh))
        result['block_auc'] = aggregate_cycle_predictions(
            np.array(all_block), np.array(all_pids), np.array(all_labels_block))
        result['rhythm_acc'] = np.mean(np.array(all_labels_rhythm) ==
                                        np.argmax(all_rhythm, axis=1))

    return result['acs_auc']


def train_multimodal_full(model, train_loader, val_loader, config, model_name='multimodal', resume=False):
    """Полный цикл обучения мультимодальной модели с Early Stopping."""
    device = torch.device(config.get('device', 'cpu'))
    model = model.to(device)

    start_epoch = 0
    best_auc = 0.0
    patience_counter = 0

    if resume:
        ckpt_files = sorted(Path('models/').glob(f'checkpoint_{model_name}_epoch*.pt'))
        if ckpt_files:
            latest = ckpt_files[-1]
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            start_epoch = ckpt['epoch'] + 1
            best_auc = ckpt.get('best_auc', 0.0)
            patience_counter = ckpt.get('patience', 0)
            print(f"Resumed from {latest.name} (epoch {ckpt['epoch']+1})")

    pos_weight = (len(train_loader.dataset) - np.sum(train_loader.dataset.labels)) / max(np.sum(train_loader.dataset.labels), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001),
                                  weight_decay=config.get('weight_decay', 1e-4))

    total_epochs = config.get('epochs', 50)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get('learning_rate', 0.001),
        epochs=total_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.3, div_factor=10, final_div_factor=100
    )

    if start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()

    scaler = torch.amp.GradScaler() if config.get('use_amp', False) and device.type == 'cuda' else None
    writer = SummaryWriter(log_dir=f"runs/{model_name}")

    for epoch in range(start_epoch, total_epochs):
        train_loss = train_multimodal_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()
        val_auc = validate_multimodal(model, val_loader, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Metrics/AUC', val_auc, epoch)

        print(f"Эпоха {epoch+1:2d}/{config.get('epochs', 50)} | "
              f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save encoder unconditionally after each epoch (for Colab preempt)
        torch.save(model.ecg_branch.state_dict(), f"models/{model_name}_encoder.pt")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"models/{model_name}_full.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'best_auc': best_auc, 'patience': patience_counter
            }, f"models/checkpoint_{model_name}_epoch{epoch+1}.pt")

        if patience_counter >= config.get('patience', 10):
            print(f"Early Stopping на эпохе {epoch+1}")
            break

    checkpoints = sorted(Path('models/').glob(f'checkpoint_{model_name}_epoch*.pt'))
    for ckpt in checkpoints[:-2]:
        ckpt.unlink()

    writer.close()
    print("=" * 60)
    print(f" ОБУЧЕНИЕ ЗАВЕРШЕНО! AUC: {best_auc:.4f}")
    print("=" * 60)
    return best_auc
