# ============================================================
# ACS ECG Detector  Grad-CAM for 1D CNN
# ============================================================

import torch
import torch.nn as nn
import numpy as np


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Находит последний Conv1d в энкодере (устойчиво к изменениям архитектуры)."""
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv1d):
            return m
    raise ValueError("Conv1d layer not found in model")


def grad_cam_1d(model: nn.Module, ecg_input: torch.Tensor) -> np.ndarray:
    """
    Grad-CAM для 1D-сигналов.
    ecg_input: [1, 12, 350]
    Возвращает: карта активации [12, 350]
    """
    model.eval()
    
    target_layer = find_last_conv_layer(model)
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(ecg_input)
    model.zero_grad()
    output.backward()
    
    weights = torch.mean(gradients, dim=(2))  # [1, C]
    cam = torch.zeros(activations.shape[2:])   # [1, C, L]  [L]
    
    for k in range(weights.shape[1]):
        cam += weights[0, k] * activations[0, k, :]
    
    cam = torch.relu(cam).cpu().numpy()
    
    # Upsample to 12 leads  350 samples
    cam = np.tile(cam, (12, 1))
    cam = cam / (cam.max() + 1e-8)
    
    forward_handle.remove()
    backward_handle.remove()
    
    return cam
