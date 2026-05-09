"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Reference: Chen et al., 2020 (https://arxiv.org/abs/2002.05709)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR self-supervised learning framework.

    Args:
        backbone: Encoder backbone name ('resnet18', 'resnet50')
        projection_dim: Output dimension of projection head
        temperature: NT-Xent loss temperature
        pretrained: Whether to initialize backbone with ImageNet weights
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        projection_dim: int = 128,
        temperature: float = 0.07,
        pretrained: bool = False,
    ):
        super().__init__()
        self.temperature = temperature

        # Build encoder
        encoder_fn = getattr(models, backbone)
        encoder = encoder_fn(pretrained=pretrained)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()  # Remove classification head
        self.encoder = encoder

        # Projection head
        self.projector = ProjectionHead(
            in_dim=feature_dim,
            hidden_dim=feature_dim,
            out_dim=projection_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized feature embeddings (for downstream tasks)."""
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass with two augmented views.
        Returns NT-Xent loss.
        """
        z1 = F.normalize(self.projector(self.encoder(x1)), dim=-1)
        z2 = F.normalize(self.projector(self.encoder(x2)), dim=-1)
        loss = self.nt_xent_loss(z1, z2)
        return loss

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss."""
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        # Similarity matrix
        sim = torch.mm(z, z.T) / self.temperature  # [2N, 2N]

        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N),
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss
