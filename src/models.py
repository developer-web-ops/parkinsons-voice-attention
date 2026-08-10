"""PyTorch models: an MLP baseline and a group-attention network."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLPBaseline(nn.Module):
    """Plain feed-forward network over the full standardised feature vector."""

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (256, 64), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class GroupAttentionNet(nn.Module):
    """Encodes each acoustic feature block separately and fuses them by attention.

    Every block (baseline jitter/shimmer, MFCC, wavelet, TQWT, ...) is projected to
    a shared embedding size, then a learned additive-attention head produces one
    weight per block. The weights are returned alongside the logit and are the
    model's built-in explanation of which acoustic families drove a prediction.
    """

    def __init__(
        self,
        group_sizes: list[int],
        embed_dim: int = 64,
        attn_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.group_sizes = group_sizes
        self.encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(size, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
            )
            for size in group_sizes
        )
        self.attn = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def embed(self, x: Tensor) -> tuple[Tensor, Tensor]:
        chunks = torch.split(x, self.group_sizes, dim=1)
        encoded = [enc(c) for enc, c in zip(self.encoders, chunks, strict=True)]
        embeddings = torch.stack(encoded, dim=1)
        weights = torch.softmax(self.attn(embeddings).squeeze(-1), dim=1)
        return embeddings, weights

    def forward(self, x: Tensor) -> Tensor:
        embeddings, weights = self.embed(x)
        context = torch.einsum("bg,bgd->bd", weights, embeddings)
        return self.head(context).squeeze(-1)

    def forward_with_attention(self, x: Tensor) -> tuple[Tensor, Tensor]:
        embeddings, weights = self.embed(x)
        context = torch.einsum("bg,bgd->bd", weights, embeddings)
        return self.head(context).squeeze(-1), weights


class LogitWrapper(nn.Module):
    """Keeps the logit two-dimensional, as SHAP's gradient explainers require."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).unsqueeze(-1)
