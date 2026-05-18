#!/usr/bin/env python
"""Обучает модель классификации ритма (ResNet1D encoder frozen + новая голова).
3 класса: 0=SR, 1=AFIB/AFLT, 2=OTHER/PACE.
Всего 768 обучаемых параметров.
"""
import torch, torch.nn as nn, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loader import create_dataloaders
from src.models.cnn_model import ResNet1D

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Load encoder (frozen)
print("Loading encoder...")
model = ResNet1D(dropout=0.3)
enc = torch.load("models/resnet1d_encoder.pt", map_location=device)
try:
    model.load_state_dict(enc, strict=False)
except:
    model.get_encoder().load_state_dict(enc)

for p in model.stem.parameters(): p.requires_grad = False
for p in model.resblock1.parameters(): p.requires_grad = False
for p in model.resblock2.parameters(): p.requires_grad = False
for p in model.resblock3.parameters(): p.requires_grad = False
for p in model.global_pool.parameters(): p.requires_grad = False

model.fc = nn.Linear(256, 3)
model = model.to(device)
trainable = sum(p.numel() for p in model.fc.parameters())
print(f"Trainable params: {trainable}")

# 2. Custom dataset that returns rhythm3 labels
class RhythmDataset:
    def __init__(self, split="train"):
        from src.data.loader import ECGDataset
        self.ecg = ECGDataset(split=split)
        self.y3 = np.load(f"data/processed/y_rhythm3_{split}.npy")
    def __len__(self):
        return len(self.ecg)
    def __getitem__(self, idx):
        x, _, pid = self.ecg[idx]
        return x, self.y3[idx], pid
    @property
    def labels(self):
        return self.y3

# 3. Data
print("Loading data...")
train_ds = RhythmDataset("train")
val_ds = RhythmDataset("val")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, num_workers=2)

# 4. Train
opt = torch.optim.Adam(model.fc.parameters(), lr=0.01)
cri = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(10):
    model.train()
    tl = 0
    for bx, by, _ in train_loader:
        bx, by = bx.to(device), by.to(device).long()
        opt.zero_grad()
        out = model(bx)
        loss = cri(out, by)
        loss.backward()
        opt.step()
        tl += loss.item()
    
    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by, _ in val_loader:
            out = model(bx.to(device))
            pred = out.argmax(dim=1).cpu()
            correct += (pred == by).sum().item()
            total += len(by)
    acc = correct / total
    print(f"Epoch {epoch+1}: loss={tl/len(train_loader):.4f} val_acc={acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/rhythm_model.pt")
        print(f"  -> saved (acc={acc:.4f})")

ckpt = torch.load("models/rhythm_model.pt", map_location="cpu")
print(f"\nDone. Best acc: {best_acc:.4f}")
print(f"FC weight shape: {ckpt['fc.weight'].shape}")
