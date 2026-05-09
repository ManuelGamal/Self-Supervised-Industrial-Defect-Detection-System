"""
Anomaly Detection via feature distance from normal distribution.
Uses pretrained SSL encoder to extract features and scores anomalies
using k-NN distance or Mahalanobis distance.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torchvision import models
from typing import Optional


class AnomalyDetector(nn.Module):
    """
    Anomaly detector using pretrained encoder + k-NN distance scoring.

    During fit(): extracts embeddings from normal training images.
    During predict(): scores new images by distance to normal embeddings.
    """

    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        backbone: str = "resnet50",
        k: int = 5,
        device: str = "cuda",
    ):
        super().__init__()
        self.k = k
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build encoder
        encoder_fn = getattr(models, backbone)
        self.encoder = encoder_fn(pretrained=False)
        self.encoder.fc = nn.Identity()

        if encoder_checkpoint:
            state_dict = torch.load(encoder_checkpoint, map_location=self.device)
            self.encoder.load_state_dict(state_dict)
            print(f"Loaded encoder from: {encoder_checkpoint}")

        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        self.nn_model = None
        self.normal_embeddings = None

    @torch.no_grad()
    def extract_embeddings(self, loader) -> np.ndarray:
        """Extract L2-normalized feature embeddings from a DataLoader."""
        all_embeddings = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)
            emb = self.encoder(x)
            emb = nn.functional.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def fit(self, normal_loader):
        """Fit k-NN index on normal image embeddings."""
        print("Extracting normal embeddings...")
        self.normal_embeddings = self.extract_embeddings(normal_loader)

        print(f"Fitting k-NN (k={self.k}) on {len(self.normal_embeddings)} normal samples...")
        self.nn_model = NearestNeighbors(n_neighbors=self.k, metric="cosine", n_jobs=-1)
        self.nn_model.fit(self.normal_embeddings)
        print("Anomaly detector fitted.")

    def anomaly_score(self, loader) -> np.ndarray:
        """
        Compute anomaly score for each image.
        Score = mean distance to k nearest normal neighbors.
        Higher score → more anomalous.
        """
        assert self.nn_model is not None, "Call fit() before predict()."
        embeddings = self.extract_embeddings(loader)
        distances, _ = self.nn_model.kneighbors(embeddings)
        scores = distances.mean(axis=1)
        return scores

    def predict(self, loader, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 0=normal, 1=defect."""
        scores = self.anomaly_score(loader)
        return (scores > threshold).astype(int)


class UNetDecoder(nn.Module):
    """
    Lightweight U-Net decoder for pixel-wise defect segmentation.
    Attaches to ResNet encoder skip connections.
    """

    def __init__(self, encoder_channels=(64, 256, 512, 1024, 2048), num_classes=1):
        super().__init__()
        c = encoder_channels

        self.up4 = self._up_block(c[4] + c[3], 512)
        self.up3 = self._up_block(512 + c[2], 256)
        self.up2 = self._up_block(256 + c[1], 128)
        self.up1 = self._up_block(128 + c[0], 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, skips):
        """skips: list of feature maps from ResNet stages [s1, s2, s3, s4, s5]"""
        s1, s2, s3, s4, s5 = skips

        if s5.shape[-2:] != s4.shape[-2:]:
            s5 = nn.functional.interpolate(
                s5, size=s4.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.up4(torch.cat([s5, s4], dim=1))

        if x.shape[-2:] != s3.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=s3.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.up3(torch.cat([x, s3], dim=1))

        if x.shape[-2:] != s2.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=s2.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.up2(torch.cat([x, s2], dim=1))

        if x.shape[-2:] != s1.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=s1.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.up1(torch.cat([x, s1], dim=1))
        return self.head(x)
