"""Hydra + PyTorch Lightning entrypoint for supervised training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from src.data.datamodule import MVTecDataModule
from src.models.lit_module import DefectClassifier


def _abs_path(path: str) -> Path:
    return Path(to_absolute_path(path)).resolve()


def _optional_abs_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    path = str(path)
    if not path.strip():
        return None
    return _abs_path(path)


def _run_name(cfg: DictConfig) -> str:
    fold_suffix = ""
    if cfg.data.get("fold_id") is not None:
        fold_suffix = f"-fold{cfg.data.fold_id}"
    return f"{cfg.experiment.name}-{cfg.data.category}-r{cfg.data.ratio}{fold_suffix}"


def _run_output_dir(cfg: DictConfig) -> Path:
    root = _abs_path(cfg.experiment.output_root)
    out = root / cfg.data.category / f"ratio{cfg.data.ratio}"
    if cfg.data.get("fold_id") is not None:
        out = out / f"fold{cfg.data.fold_id}"
    return out


def _checkpoint_dir(cfg: DictConfig) -> Path:
    base = _abs_path(cfg.checkpoint.dir)
    out = base / cfg.data.category / f"ratio{cfg.data.ratio}"
    if cfg.data.get("fold_id") is not None:
        out = out / f"fold{cfg.data.fold_id}"
    return out


def _latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def _resolve_resume_checkpoint(cfg: DictConfig, ckpt_dir: Path) -> Optional[str]:
    configured = cfg.resume.checkpoint_path
    if configured:
        explicit = _abs_path(configured)
        if explicit.exists():
            return str(explicit)
        raise FileNotFoundError(f"Configured resume checkpoint not found: {explicit}")

    if cfg.resume.auto:
        ckpt = _latest_ckpt(ckpt_dir)
        if ckpt is not None:
            return str(ckpt)

    return None


def _effective_precision(cfg: DictConfig) -> str:
    precision = str(cfg.training.precision)
    accelerator = str(cfg.training.accelerator)
    if precision == "16-mixed" and accelerator == "cpu":
        return "32-true"
    if precision == "16-mixed" and accelerator == "auto" and not torch.cuda.is_available():
        return "32-true"
    return precision


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.experiment.seed), workers=True)

    run_output_dir = _run_output_dir(cfg)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MVTecDataModule(
        root=_abs_path(cfg.data.root),
        category=str(cfg.data.category),
        ratio=int(cfg.data.ratio),
        split_csv=_optional_abs_path(cfg.data.get("split_csv")),
        splits_dir=_abs_path(cfg.data.splits_dir),
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        image_size=int(cfg.data.image_size),
    )

    model = DefectClassifier(
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        epochs=int(cfg.training.epochs),
        warmup_epochs=int(cfg.training.warmup_epochs),
        focal_gamma=float(cfg.model.focal_gamma),
        focal_alpha=float(cfg.model.focal_alpha),
        dropout=float(cfg.model.dropout),
        pretrained=bool(cfg.model.pretrained),
    )

    if bool(cfg.model.freeze_encoder):
        for param in model.encoder.parameters():
            param.requires_grad = False

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch={epoch:03d}-val_auroc={val/auroc:.4f}",
            monitor=str(cfg.checkpoint.monitor),
            mode=str(cfg.checkpoint.mode),
            save_top_k=int(cfg.checkpoint.save_top_k),
            save_last=bool(cfg.checkpoint.save_last),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if bool(cfg.early_stopping.enabled):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.early_stopping.monitor),
                mode=str(cfg.early_stopping.mode),
                patience=int(cfg.early_stopping.patience),
                min_delta=float(cfg.early_stopping.min_delta),
            )
        )

    logger = None
    if bool(cfg.wandb.enabled):
        logger = WandbLogger(
            entity=str(cfg.wandb.entity),
            project=str(cfg.wandb.project),
            name=_run_name(cfg),
            group=str(cfg.wandb.group),
            tags=list(cfg.wandb.tags),
            notes=str(cfg.wandb.notes),
            save_dir=str(run_output_dir),
            log_model=cfg.wandb.log_model,
        )
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        logger=logger,
        accelerator=str(cfg.training.accelerator),
        devices=cfg.training.devices,
        precision=_effective_precision(cfg),
        callbacks=callbacks,
        deterministic=bool(cfg.training.deterministic),
        benchmark=bool(cfg.training.benchmark),
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        default_root_dir=str(run_output_dir),
    )

    resume_ckpt = _resolve_resume_checkpoint(cfg, ckpt_dir)
    if resume_ckpt:
        print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")
    else:
        print("[INFO] Starting a fresh training run (no checkpoint resume).")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt)

    if bool(cfg.evaluation.run_test_after_fit):
        test_ckpt = resume_ckpt
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                test_ckpt = cb.best_model_path
                break
        trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt or None)

    summary = {
        "run_name": _run_name(cfg),
        "category": str(cfg.data.category),
        "ratio": int(cfg.data.ratio),
        "fold_id": cfg.data.get("fold_id"),
        "split_csv": str(_optional_abs_path(cfg.data.get("split_csv")) or ""),
        "best_model_path": "",
        "best_model_score": None,
        "resumed_from": resume_ckpt,
    }
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            summary["best_model_path"] = cb.best_model_path
            if cb.best_model_score is not None:
                summary["best_model_score"] = float(cb.best_model_score.cpu().item())
            break

    summary_path = run_output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote run summary: {summary_path}")

    if logger is not None:
        logger.experiment.finish()


if __name__ == "__main__":
    main()
