"""A deliberately small audio CNN for the Phase 2D comparison.

The MDVR-KCL corpus has only 37 subjects, so the spec mandates a *small* model and
forbids large pretrained backbones (wav2vec2 / WavLM are explicitly out of scope
here). This is a compact 3-block 2-D CNN over log-Mel windows with global average
pooling and a single logit head — on the order of a few thousand parameters, which
is appropriate for the data budget and keeps CPU training/inference cheap.

Architecture (input ``(N, 1, n_mels, n_frames)``):

    [Conv3x3 -> BN -> ReLU -> MaxPool2] x 2   (channels 1->8->16)
    Conv3x3 -> BN -> ReLU                     (channels 16->32)
    AdaptiveAvgPool2d(1) -> Dropout -> Linear(32 -> 1)

Global average pooling makes the head independent of the exact time/frequency
extent, and the low channel counts keep the parameter count tiny.
"""

from __future__ import annotations

import torch
from torch import nn

from src.audio.cnn_config import DEFAULT_CNN_CONFIG, CnnConfig


class SmallAudioCNN(nn.Module):
    """Compact log-Mel CNN emitting a single logit per window."""

    def __init__(self, config: CnnConfig = DEFAULT_CNN_CONFIG) -> None:
        super().__init__()
        c1, c2, c3 = config.conv_channels
        k, pad = config.kernel_size, config.kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, k, padding=pad),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, k, padding=pad),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, k, padding=pad),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(c3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)  # (N, c3)
        x = self.dropout(x)
        return self.head(x)  # (N, 1) logits


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
