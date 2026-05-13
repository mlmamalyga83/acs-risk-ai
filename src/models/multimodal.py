# ============================================================
# ACS ECG Detector — multimodal model (ECG + clinical)
# v26.0 — 4 heads: ACS, LVH, Block, Rhythm (7-class)
# ============================================================

import torch
import torch.nn as nn


class MultimodalECGNet(nn.Module):
    """ECG CNN-encoder + clinical FFN -> 4 clinical heads."""
    
    def __init__(self, cnn_encoder, clinical_dim=2, embedding_dim=256):
        super().__init__()
        self.ecg_branch = cnn_encoder
        
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        concat_dim = embedding_dim + 32
        
        # 4 clinical heads
        self.head_acs = nn.Sequential(
            nn.Linear(concat_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )
        self.head_glzh = nn.Sequential(
            nn.Linear(concat_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )
        self.head_block = nn.Sequential(
            nn.Linear(concat_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )
        self.head_rhythm = nn.Sequential(
            nn.Linear(concat_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 7)
        )
    
    def forward(self, ecg, clinical):
        ecg_feat = self.ecg_branch(ecg).view(ecg.shape[0], -1)
        clin_feat = self.clinical_branch(clinical)
        combined = torch.cat([ecg_feat, clin_feat], dim=1)
        
        return {
            'acs':    self.head_acs(combined).squeeze(-1),
            'glzh':   self.head_glzh(combined).squeeze(-1),
            'block':  self.head_block(combined).squeeze(-1),
            'rhythm': self.head_rhythm(combined)
        }
    
    def freeze_encoder(self):
        for param in self.ecg_branch.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.ecg_branch.parameters():
            param.requires_grad = True
