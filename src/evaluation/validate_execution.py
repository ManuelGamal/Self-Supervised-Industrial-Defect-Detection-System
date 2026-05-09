"""Validate Engineer-4 execution criteria: 6 categories × 3 folds, 100% labels, 18 W&B runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import wandb


# Run naming convention set by Engineer 3
RUN_NAME_PATTERN = re.compile(r"^supervised-defect-(?P<category>[^-]+)-r100-fold(?P<fold>\d+)$")

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


def _pick(flat: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in flat and flat[key] is not None:
            return flat[key]
    return None


def _parse_run_name(run_name: str) -> tuple[str | None, int | None]:
    """Extract (category, fold) from supervised-defect-{category}-r100-fold{N}."""
    m = RUN_NAME_PATTERN.match(run_name)
    if m is None:
        return None, None
    return m.group("category"), int(m.group("fold"))


def _epochs(flat_cfg: dict[str, Any]) -> int | None:
    for key in ("training.epochs", "epochs", "cfg.training.epochs", "cfg.epochs"):
        iv = _as_int(flat_cfg.get(key))
        if iv is not None:
            return iv
    return None


def _is_final_epoch(epoch_value: Any, configured_epochs: int | None) -> bool:
    epoch = _as_int(epoch_value)
    if epoch is None:
        return False
    if configured_epochs is None:
        return True
    return epoch in {configured_epochs - 1, configured_epochs}


def _load_manifest(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x).replace("\\", "/") for x in data}
        if isinstance(data, dict):
            if "files" in data and isinstance(data["files"], list):
                return {str(x).replace("\\", "/") for x in data["files"]}
            raise ValueError("JSON manifest must be list[str] or {'files': list[str]}.")
        raise ValueError("Unsupported JSON manifest schema.")

    if suffix == ".csv":
        paths: set[str] = set()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            preferred = None
            for candidate in ("path", "file", "name", "filepath"):
                if candidate in cols:
                    preferred = candidate
                    break
            if preferred is None:
                raise ValueError(
                    f"CSV manifest must include one of columns: path,file,name,filepath. Got: {sorted(cols)}"
                )
            for row in reader:
                if row.get(preferred):
                    paths.add(str(row[preferred]).replace("\\", "/"))
        return paths

    # Plain text fallback: one path per line
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Engineer-4 run execution criteria. "
            "Expects 18 finished W&B runs: 6 categories × 3 folds at 100%% labels. "
            "Run naming convention: supervised-defect-{category}-r100-fold{N}."
        )
    )
    parser.add_argument("--entity", type=str, default="manuelaziz27-ain-shams-university")
    parser.add_argument("--project", type=str, default="defect-detection-supervised")
    parser.add_argument("--group",   type=str, default="")
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=18,
        help="Expected number of finished runs (6 categories × 3 folds = 18).",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=str,
        required=True,
        help="Path to Kaggle file listing (json/csv/txt).",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default="{category}_r100_f{fold}_best.ckpt",
        help="Expected checkpoint filename template (matches Engineer 3 naming).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    if args.group:
        runs = [r for r in runs if r.group == args.group]

    if len(runs) != args.expected_runs:
        raise RuntimeError(f"Expected {args.expected_runs} runs, found {len(runs)}.")

    bad_states = [r for r in runs if r.state != "finished"]
    if bad_states:
        raise RuntimeError(
            "Found non-finished runs: "
            + ", ".join(f"{r.name}:{r.state}" for r in bad_states)
        )

    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise RuntimeError("Run names are not unique.")

    name_mismatch: list[str] = []
    missing_metrics: list[str] = []
    bad_epoch: list[str] = []
    expected_checkpoints: set[str] = set()
    seen_combos: set[tuple[str, int]] = set()
    duplicate_combos: list[str] = []

    for run in runs:
        flat_cfg: dict[str, Any] = {}
        flat_summary: dict[str, Any] = {}
        _flatten_dict("", dict(run.config), flat_cfg)
        _flatten_dict("", dict(run.summary), flat_summary)

        category, fold = _parse_run_name(run.name)
        if category is None or fold is None:
            name_mismatch.append(
                f"{run.name!r} does not match supervised-defect-{{category}}-r100-fold{{N}}"
            )
            continue

        combo = (category, fold)
        if combo in seen_combos:
            duplicate_combos.append(f"{category} fold{fold} ({run.name})")
        seen_combos.add(combo)

        configured_epochs = _epochs(flat_cfg)

        val_auroc  = _pick(flat_summary, ("val/auroc",  "summary.val/auroc"))
        val_loss   = _pick(flat_summary, ("val/loss",   "summary.val/loss"))
        train_loss = _pick(flat_summary, ("train/loss_epoch", "train/loss", "train_loss"))
        epoch      = _pick(flat_summary, ("epoch",))

        if any(v is None for v in (val_auroc, val_loss, train_loss, epoch)):
            missing_metrics.append(run.name)
        elif not _is_final_epoch(epoch, configured_epochs):
            bad_epoch.append(
                f"{run.name} epoch={epoch} configured_epochs={configured_epochs}"
            )

        expected_checkpoints.add(
            args.checkpoint_template.format(category=category, fold=fold).replace("\\", "/")
        )

    # --- Validate complete coverage ---

    missing_combos = [
        f"{cat} fold{f}"
        for cat in sorted(CATEGORIES)
        for f in sorted(FOLDS)
        if (cat, f) not in seen_combos
    ]

    # --- Report all errors ---

    if name_mismatch:
        raise RuntimeError(
            "Run names do not match required naming convention "
            "(supervised-defect-{category}-r100-fold{N}): "
            + "; ".join(name_mismatch[:20])
        )

    if duplicate_combos:
        raise RuntimeError(
            f"Duplicate (category, fold) combinations found: {duplicate_combos}"
        )

    if missing_combos:
        raise RuntimeError(
            f"Missing expected (category, fold) combinations: {missing_combos}"
        )

    if missing_metrics:
        raise RuntimeError(
            "Some runs are missing required final metrics (val_auroc, val_loss, train_loss, epoch): "
            + ", ".join(missing_metrics[:20])
        )

    if bad_epoch:
        raise RuntimeError(
            "Some runs did not log metrics at final epoch: " + "; ".join(bad_epoch[:20])
        )

    manifest = _load_manifest(Path(args.checkpoint_manifest))
    missing_checkpoints = sorted(expected_checkpoints - manifest)
    if missing_checkpoints:
        raise RuntimeError(
            "Checkpoint manifest is missing expected files: "
            + ", ".join(missing_checkpoints[:20])
        )

    print("[OK] Run-state, naming, metrics, and checkpoint-manifest checks passed.")
    print(f"[OK] Finished runs: {len(runs)}")
    print(f"[OK] Coverage: {len(seen_combos)} (category, fold) combinations verified.")
    print(f"[OK] Checked checkpoints: {len(expected_checkpoints)}")


if __name__ == "__main__":
    main()
