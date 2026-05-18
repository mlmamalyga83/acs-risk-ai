#!/usr/bin/env python
"""Переобучает только FC-слой модели ResNet1D с правильной инициализацией.
Encoder заморожен — обучается только 257 параметров (256 weights + 1 bias).
"""
import sys, torch, torch.nn as nn, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sklearn.metrics import roc_auc_score


def main():
    device = "cpu"
    print("Loading encoder...")
    from src.data.loader import create_dataloaders
    from src.models.cnn_model import ResNet1D

    model = ResNet1D(dropout=0.3)

    encoder_ckpt = torch.load("models/resnet1d_encoder.pt", map_location=device)
    model.get_encoder().load_state_dict(encoder_ckpt)

    # Freeze encoder
    for param in model.stem.parameters():
        param.requires_grad = False
    for param in model.resblock1.parameters():
        param.requires_grad = False
    for param in model.resblock2.parameters():
        param.requires_grad = False
    for param in model.resblock3.parameters():
        param.requires_grad = False
    for param in model.global_pool.parameters():
        param.requires_grad = False

    # Reinit FC layer with proper weights
    model.fc = nn.Linear(256, 1)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, -2.0)

    model = model.to(device)
    trainable = sum(p.numel() for p in model.fc.parameters())
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    print(f"Trainable: {trainable} (FC layer), Frozen: {frozen}, Total: {total}")

    # Data - use subset for speed on CPU
    pp = "data/processed/"
    print("Loading data...")
    train_loader = create_dataloaders(split="train", batch_size=256, processed_path=pp)
    val_loader = create_dataloaders(split="val", batch_size=256, processed_path=pp)

    # Use first 10000 batches of train for speed
    train_batches = []
    for i, batch in enumerate(train_loader):
        if i >= 3000:
            break
        train_batches.append(batch)

    # Optimizer (only FC)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))

    best_auc = 0.0
    for epoch in range(5):
        # Train
        model.train()
        total_loss = 0
        for batch_x, batch_y, _ in train_batches:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        all_probas, all_labels, all_pids = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_pid in val_loader:
                batch_x = batch_x.to(device)
                outputs = torch.sigmoid(model(batch_x))
                all_probas.extend(outputs.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_pids.extend(batch_pid.numpy())

        from src.train.trainer import aggregate_cycle_predictions
        val_auc = aggregate_cycle_predictions(
            np.array(all_probas), np.array(all_pids), np.array(all_labels)
        )
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:2d}/10 | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "models/resnet1d_full.pt")
            torch.save(model.get_encoder().state_dict(), "models/resnet1d_encoder.pt")
            print(f"  -> Saved (AUC={val_auc:.4f})")

    print(f"\nDone. Best AUC: {best_auc:.4f}")
    print(f"Models saved to models/resnet1d_full.pt and models/resnet1d_encoder.pt")

    # Verify FC weights
    ckpt = torch.load("models/resnet1d_full.pt", map_location="cpu")
    print(f"FC weight mean={ckpt['fc.weight'].mean().item():.4f} "
          f"std={ckpt['fc.weight'].std().item():.4f} "
          f"bias={ckpt['fc.bias'].item():.4f}")


if __name__ == "__main__":
    main()
