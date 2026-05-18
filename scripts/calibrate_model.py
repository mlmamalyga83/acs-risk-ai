#!/usr/bin/env python
"""Применяет Temperature Scaling (T=4.50) к модели ResNet1D.
Не меняет AUC, но улучшает калибровку (Brier Score).
"""
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cnn_model import ResNet1D

T = 4.50

print("Loading model...")
model = ResNet1D(dropout=0.3)
ckpt = torch.load("models/resnet1d_full.pt", map_location="cpu")
model.load_state_dict(ckpt)

print("FC weight mean before:", ckpt["fc.weight"].mean().item() )
print("FC bias before:", ckpt["fc.bias"].item())

with torch.no_grad():
    model.fc.weight.data /= T
    model.fc.bias.data /= T

print("FC weight mean after: ", model.fc.weight.mean().item())
print("FC bias after: ", model.fc.bias.item())

torch.save(model.state_dict(), "models/resnet1d_full_calibrated.pt")
torch.save(model.get_encoder().state_dict(), "models/resnet1d_encoder_calibrated.pt")
print("Saved models/resnet1d_full_calibrated.pt")
print("Done")
