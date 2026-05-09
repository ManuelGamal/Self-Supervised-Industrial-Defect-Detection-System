"""
SSL Pretraining Script
Supports SimCLR, BYOL, DINO methods.

Usage:
    python src/ssl/train_ssl.py --config configs/ssl_pretrain.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import IndustrialDataset, SSLTransform
from src.ssl.simclr import SimCLR


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for (x1, x2) in tqdm(loader, desc="Training", leave=False):
        x1, x2 = x1.to(device), x2.to(device)
        loss = model(x1, x2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def linear_probe_eval(encoder, train_loader, val_loader, feature_dim, num_classes, device):
    """Quick linear evaluation to monitor SSL representation quality."""
    probe = nn.Linear(feature_dim, num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    encoder.eval()
    for epoch in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = encoder(x)
            loss = nn.CrossEntropyLoss()(probe(feats), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            feats = encoder(x)
            preds = probe(feats).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    ssl_transform = SSLTransform(
        image_size=config.get("image_size", 224),
        strength=config.get("color_jitter_strength", 0.5),
    )
    dataset = IndustrialDataset(
        root_dir=config["data_dir"],
        split="train",
        mode="ssl",
        transform=ssl_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=config.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )

    # Model
    method = config.get("method", "simclr")
    if method == "simclr":
        model = SimCLR(
            backbone=config.get("backbone", "resnet50"),
            projection_dim=config.get("projection_dim", 128),
            temperature=config.get("temperature", 0.07),
        ).to(device)
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented. Choices: simclr")

    # Optimizer with cosine schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 3e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 200)
    )

    # Training loop
    save_dir = Path(config.get("save_dir", "checkpoints/ssl"))
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.get("epochs", 200) + 1):
        loss = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        print(f"Epoch [{epoch}/{config['epochs']}] | Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % config.get("save_every_n_epochs", 20) == 0:
            ckpt_path = save_dir / f"encoder_epoch{epoch}.pth"
            torch.save(model.encoder.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final encoder
    torch.save(model.encoder.state_dict(), save_dir / "best_encoder.pth")
    print("Training complete. Final encoder saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ssl_pretrain.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    flat_config = {**config.get("ssl", {}), **config.get("training", {}),
                   **config.get("data", {}), **config.get("checkpoint", {})}
    main(flat_config)
