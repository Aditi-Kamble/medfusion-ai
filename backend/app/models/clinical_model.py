"""
Branch 1 of MedFusion AI: a feed-forward neural network that processes
clinical/tabular patient data and outputs a learned feature representation
(clinical_features) plus, standalone, a risk probability.

In the fused model (built later), we'll reuse `clinical_features` — the
second-to-last layer's output — as input to the fusion layer, instead of
the final sigmoid output.
"""

import torch
import torch.nn as nn


class ClinicalNet(nn.Module):
    def __init__(self, input_dim=13, hidden_dims=(64, 32), feature_dim=16, dropout=0.3):
        super(ClinicalNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], feature_dim),
            nn.ReLU(),
        )

        # Standalone classifier head — used when training/evaluating Branch 1 alone
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        if return_features:
            return features
        out = self.classifier(features)
        return torch.sigmoid(out)