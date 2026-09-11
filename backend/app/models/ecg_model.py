"""
Branch 2 of MedFusion AI: a CNN that processes ECG images and outputs a
learned feature representation (ecg_features), plus, standalone, class
probabilities for 4 categories (Normal, Abnormal Heartbeat, History of MI, MI).

Same design principle as ClinicalNet: the feature_extractor's output is what
gets reused in the fusion model later.
"""

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    def __init__(self, num_classes=4, feature_dim=32, dropout=0.4):
        super(ECGNet, self).__init__()

        self.conv_block = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Global average pool -> 128 x 1 x 1
            nn.AdaptiveAvgPool2d(1),
        )

        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Standalone classifier head — used when training/evaluating Branch 2 alone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv_block(x)
        features = self.feature_head(x)
        if return_features:
            return features
        out = self.classifier(features)
        return out  # raw logits — use CrossEntropyLoss (applies softmax internally)