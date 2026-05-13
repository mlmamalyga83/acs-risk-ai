# ============================================================
# ACS ECG Detector  CNN architectures (Simple1DCNN, ResNet1D)
# ============================================================

import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    """Baseline CNN: 3Conv1d  MaxPool  Flatten  FC."""
    
    def __init__(self, input_channels=12, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)
    
    def get_encoder(self):
        """Возвращает все слои КРОМЕ последнего Linear (классификационной головы)."""
        modules = list(self.children())[:-1]
        return nn.Sequential(*modules)


class ResBlock1D(nn.Module):
    """Residual block for 1D signals."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if stride == 2 else None
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.pool(out)
        return out


class ResNet1D(nn.Module):
    """ResNet1D for ECG classification."""
    
    def __init__(self, input_channels=12, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.resblock1 = ResBlock1D(64, 64, kernel_size=7)
        self.resblock2 = ResBlock1D(64, 128, kernel_size=5)
        self.resblock3 = ResBlock1D(128, 256, kernel_size=3)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)
    
    def get_encoder(self):
        """Возвращает все слои КРОМЕ последнего Linear."""
        modules = [self.stem, self.resblock1, self.resblock2, self.resblock3, self.global_pool]
        return nn.Sequential(*modules)
