# Fold Instability Report

Selection threshold: AUROC std dev > 0.02 is flagged as unstable.
Metric source: `val_auroc`.

## Per-category consistency

| Category | Mean AUROC | AUROC Std | Mean F1 | Folds | Unstable |
|---|---:|---:|---:|---:|---|
| bottle | 0.9860 | 0.0116 | 0.6796 | 3 | No |
| capsule | 0.9395 | 0.0253 | 0.7564 | 3 | Yes |
| carpet | 0.9393 | 0.0267 | 0.7109 | 3 | Yes |
| hazelnut | 0.9843 | 0.0119 | 0.8067 | 3 | No |
| leather | 0.9985 | 0.0021 | 0.9190 | 3 | No |
| pill | 0.8864 | 0.0256 | 0.7365 | 3 | Yes |

## Flagged unstable categories

- capsule (std 0.0253)
- carpet (std 0.0267)
- pill (std 0.0256)

## Notes

- This report is computed from `C:/Users/acer/Downloads/results (1).parquet`.
- Best-checkpoint selection is now correctly based on `val_auroc` with `val_f1` tie-break.
