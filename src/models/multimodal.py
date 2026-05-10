# ============================================================
# ACS ECG Detector — multimodal model (ECG + clinical)
# ============================================================

import torch
import torch.nn as nn


class MultimodalECGNet(nn.Module):
    """ECG CNN-encoder + clinical FFN → concatenation → classifier."""
    
    def __init__(self, cnn_encoder, clinical_dim=5, embedding_dim=128):
        super().__init__()
        self.ecg_branch = cnn_encoder
        
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        
        concat_dim = embedding_dim + 32
        
        # Main head: ACS prediction
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # Auxiliary heads (×0.1 weight in loss)
        self.head_glzh = nn.Linear(concat_dim, 1)      # LVH
        self.head_block = nn.Linear(concat_dim, 1)     # Block
        self.head_norm = nn.Linear(concat_dim, 1)      # Normal
    
    def forward(self, ecg, clinical):
        ecg_feat = self.ecg_branch(ecg).view(ecg.shape[0], -1)
        clin_feat = self.clinical_branch(clinical)
        combined = torch.cat([ecg_feat, clin_feat], dim=1)
        
        return {
            'acs': self.classifier(combined).squeeze(-1),
            'glzh': self.head_glzh(combined).squeeze(-1),
            'block': self.head_block(combined).squeeze(-1),
            'norm': self.head_norm(combined).squeeze(-1)
        }
    
    def freeze_encoder(self):
        for param in self.ecg_branch.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.ecg_branch.parameters():
            param.requires_grad = True
