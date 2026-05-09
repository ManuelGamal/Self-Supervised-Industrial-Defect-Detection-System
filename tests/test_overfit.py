"""Overfit sanity test for the supervised defect classifier."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from src.models.lit_module import DefectClassifier


def test_overfit_single_batch() -> None:
    """Model must overfit a tiny batch to near-zero loss."""
    pl.seed_everything(42)
    model = DefectClassifier(
        lr=1e-3,
        dropout=0.0,
        pretrained=False,
    )
    model.train()

    # Keep this small so it runs quickly on CPU in CI.
    x = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 2, (8,))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_loss = None
    for step in range(50):
        opt.zero_grad()
        logits = model(x)
        loss = model.loss_fn(logits, y)
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = float(loss.item())

    final_loss = float(loss.item())
    assert initial_loss is not None
    assert final_loss < initial_loss * 0.1, (
        f"Model failed to overfit: {initial_loss:.3f} -> {final_loss:.3f}"
    )

