"""Aggregate W&B runs (6 categories × 3 folds, 100% labels) into results/results.parquet."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


TRAIN_LOSS_KEYS = ("train/loss_epoch", "train/loss", "train_loss")
EPOCH_KEYS = ("epoch",)

# Run naming convention used by Engineer 3:  supervised-defect-{category}-r100-fold{fold}
RUN_NAME_PATTERN = re.compile(r"^supervised-defect-(?P<category>[^-]+)-r100-fold(?P<fold>\d+)$")

REQUIRED_OUTPUT_COLUMNS = (
    "category",
    "fold",
    "run_id",
    "run_name",
    "val_auroc",
    "val_aupr",
    "val_f1",
    "gpu_hours",
)

CATEGORIES = frozenset({"bottle", "capsule", "carpet", "hazelnut", "leather", "pill"})
FOLDS = frozenset({1, 2, 3})


def _flatten_dict(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_dict(child_prefix, child, out)
    else:
        out[prefix] = value


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick(flat: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in flat and flat[key] is not None:
            return flat[key]
    return None


def _parse_run_name(run_name: str) -> tuple[str | None, int | None]:
    """Extract (category, fold) from a run name like supervised-defect-bottle-r100-fold2."""
    m = RUN_NAME_PATTERN.match(run_name)
    if m is None:
        return None, None
    return m.group("category"), int(m.group("fold"))


def _epochs_from_config(flat_cfg: dict[str, Any]) -> int | None:
    for key in ("training.epochs", "epochs", "cfg.training.epochs", "cfg.epochs"):
        parsed = _as_int(flat_cfg.get(key))
        if parsed is not None:
            return parsed
    return None


def _epoch_is_final(epoch_value: Any, configured_epochs: int | None) -> bool:
    epoch = _as_int(epoch_value)
    if epoch is None:
        return False
    if configured_epochs is None:
        return True
    # Some loggers store epoch as 0-based, some as 1-based.
    return epoch in {configured_epochs - 1, configured_epochs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate W&B runs and enforce Engineer-4 acceptance checks. "
            "Expects 18 finished runs: 6 categories × 3 folds at 100%% labels."
        )
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="manuelaziz27-ain-shams-university",
        help="W&B entity/user/organization",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="defect-detection-supervised",
        help="W&B project name",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="",
        help="Optional run group filter (exact match).",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=18,
        help="Expected number of finished runs (6 categories × 3 folds = 18).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/results.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--allow-name-mismatch",
        action="store_true",
        help="Skip run-name convention validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api = wandb.Api(timeout=120)
    project_path = f"{args.entity}/{args.project}"
    runs = list(api.runs(project_path))

    if args.group:
        runs = [run for run in runs if run.group == args.group]

    non_finished = [run for run in runs if run.state != "finished"]
    if non_finished:
        states = ", ".join(f"{run.name}:{run.state}" for run in non_finished[:20])
        raise RuntimeError(
            "Found non-finished runs in project scope; expected all runs finished. "
            f"Examples: {states}"
        )

    if len(runs) != args.expected_runs:
        raise RuntimeError(
            f"Expected exactly {args.expected_runs} finished runs, found {len(runs)}."
        )

    rows: list[dict[str, Any]] = []
    name_mismatches: list[str] = []
    metric_failures: list[str] = []
    epoch_failures: list[str] = []

    seen_names: set[str] = set()
    duplicate_names: set[str] = set()

    seen_combos: set[tuple[str, int]] = set()
    duplicate_combos: list[str] = []

    for run in runs:
        flat_cfg: dict[str, Any] = {}
        flat_summary: dict[str, Any] = {}
        _flatten_dict("", dict(run.config), flat_cfg)
        _flatten_dict("", dict(run.summary), flat_summary)

        # Parse category and fold from the run name (ground truth naming convention).
        category, fold = _parse_run_name(run.name)
        if category is None or fold is None:
            name_mismatches.append(
                f"{run.name!r} does not match supervised-defect-{{category}}-r100-fold{{N}}"
            )
            category_for_row = None
            fold_for_row = None
        else:
            category_for_row = category
            fold_for_row = fold

            combo = (category, fold)
            if combo in seen_combos:
                duplicate_combos.append(f"{category} fold{fold} ({run.name})")
            seen_combos.add(combo)

        if run.name in seen_names:
            duplicate_names.add(run.name)
        seen_names.add(run.name)

        val_auroc = _as_float(_pick(flat_summary, ("val/auroc", "summary.val/auroc")))
        val_aupr  = _as_float(_pick(flat_summary, ("val/aupr",  "summary.val/aupr")))
        val_f1    = _as_float(_pick(flat_summary, ("val/f1",    "summary.val/f1")))
        val_loss  = _as_float(_pick(flat_summary, ("val/loss",  "summary.val/loss")))

        train_loss = _as_float(_pick(flat_summary, TRAIN_LOSS_KEYS))
        epoch      = _pick(flat_summary, EPOCH_KEYS)
        gpu_runtime_sec = _as_float(_pick(flat_summary, ("_runtime", "summary._runtime")))
        gpu_hours = (gpu_runtime_sec / 3600.0) if gpu_runtime_sec is not None else 0.0

        configured_epochs = _epochs_from_config(flat_cfg)

        missing_required = []
        if val_auroc is None:
            missing_required.append("val/auroc")
        if val_loss is None:
            missing_required.append("val/loss")
        if train_loss is None:
            missing_required.append("train/loss")
        if epoch is None:
            missing_required.append("epoch")
        if missing_required:
            metric_failures.append(f"{run.name} missing {missing_required}")

        if not _epoch_is_final(epoch, configured_epochs):
            epoch_failures.append(
                f"{run.name} epoch={epoch} configured_epochs={configured_epochs}"
            )

        row: dict[str, Any] = {
            "run_id":    run.id,
            "run_name":  run.name,
            "run_state": run.state,
            "run_url":   run.url,
            "group":     run.group,
            "job_type":  run.job_type,
            "category":  category_for_row,
            "fold":      fold_for_row,
            "val_auroc": val_auroc,
            "val_aupr":  val_aupr,
            "val_f1":    val_f1,
            "gpu_hours": gpu_hours,
            "val_loss":  val_loss,
            "train_loss": train_loss,
            "epoch":     _as_int(epoch),
        }

        for key, value in flat_cfg.items():
            row[f"cfg.{key}"] = value
        for key, value in flat_summary.items():
            row[f"summary.{key}"] = value

        rows.append(row)

    # --- Validation gates ---

    if duplicate_names:
        raise RuntimeError(f"Duplicate run names found: {sorted(duplicate_names)}")

    if duplicate_combos:
        raise RuntimeError(
            f"Duplicate (category, fold) combinations found: {duplicate_combos}"
        )

    if name_mismatches and not args.allow_name_mismatch:
        preview = "; ".join(name_mismatches[:20])
        raise RuntimeError(
            "Run name convention mismatch. Expected supervised-defect-{{category}}-r100-fold{{N}}. "
            f"Examples: {preview}"
        )

    if metric_failures:
        preview = "; ".join(metric_failures[:20])
        raise RuntimeError(f"Missing required final metrics detected: {preview}")

    if epoch_failures:
        preview = "; ".join(epoch_failures[:20])
        raise RuntimeError(f"Final epoch validation failed: {preview}")

    # --- Build DataFrame and validate schema ---

    df = pd.DataFrame(rows)
    df = df.sort_values(["category", "fold", "run_name"]).reset_index(drop=True)

    missing_output_cols = [col for col in REQUIRED_OUTPUT_COLUMNS if col not in df.columns]
    if missing_output_cols:
        raise RuntimeError(f"Output schema missing required columns: {missing_output_cols}")

    missing_val = df[df["val_auroc"].isna()]
    if not missing_val.empty:
        offenders = ", ".join(missing_val["run_name"].astype(str).tolist())
        raise RuntimeError(
            "val_auroc contains nulls. This is not allowed; fix runs before aggregation. "
            f"Offenders: {offenders}"
        )

    # Verify all 6 categories × 3 folds are present (if names parsed correctly).
    if not args.allow_name_mismatch and "category" in df.columns and "fold" in df.columns:
        valid = df.dropna(subset=["category", "fold"])
        missing_combos = [
            f"{cat} fold{fold}"
            for cat in sorted(CATEGORIES)
            for fold in sorted(FOLDS)
            if not ((valid["category"] == cat) & (valid["fold"] == fold)).any()
        ]
        if missing_combos:
            raise RuntimeError(
                f"Missing expected (category, fold) combinations: {missing_combos}"
            )

    # --- Write parquet with round-trip check ---

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Failed to write parquet. Install a parquet engine (e.g. pyarrow)."
        ) from exc

    try:
        roundtrip = pd.read_parquet(out_path)
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"Round-trip parquet read failed for {out_path} via pandas.read_parquet()."
        ) from exc
    if len(roundtrip) != len(df):
        raise RuntimeError(
            f"Round-trip row mismatch: wrote {len(df)} rows, read back {len(roundtrip)} rows."
        )

    print(f"[OK] Wrote {len(df)} rows to {out_path}")
    print(f"[OK] Required output columns present: {list(REQUIRED_OUTPUT_COLUMNS)}")
    cats = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []
    print(f"[OK] Categories: {cats}")
    folds_present = sorted(df["fold"].dropna().astype(int).unique().tolist()) if "fold" in df.columns else []
    print(f"[OK] Folds: {folds_present}")


if __name__ == "__main__":
    main()
