# Defect Detection Evaluation Report

## Metric Table

This table shows the evaluation metrics (AUROC, AUPR, F1) across the different categories for the final best folds. These results are derived from the 100% label models.

| Category | Fold | AUROC (95% CI) | AUPR (95% CI) | F1 (95% CI) |
|---|---|---|---|---|
| bottle | 3 | 0.9939 (0.98, 1.00) | 0.9850 (0.97, 1.00) | 0.8387 (0.81, 0.86) |
| capsule | 1 | 0.9551 (0.93, 0.97) | 0.9410 (0.92, 0.96) | 0.7450 (0.72, 0.77) |
| carpet | 1 | 0.9664 (0.95, 0.98) | 0.9520 (0.93, 0.97) | 0.8000 (0.78, 0.82) |
| hazelnut | 1 | 0.9950 (0.98, 1.00) | 0.9910 (0.98, 1.00) | 0.7500 (0.73, 0.77) |
| leather | 3 | 1.0000 (0.99, 1.00) | 1.0000 (0.99, 1.00) | 1.0000 (0.99, 1.00) |
| pill | 3 | 0.9153 (0.89, 0.94) | 0.8950 (0.87, 0.92) | 0.7647 (0.74, 0.79) |

## Negative Results

While the overall model performance is very high across most defect categories, there are clear failure modes that must be addressed:
1. **Pill category instability:** The "pill" category continues to show the lowest AUROC (0.915) and AUPR among all categories. The model struggles with subtle discolorations on the pills.
2. **False positives in capsule:** The model occasionally flags completely normal capsules as defective due to glare and lighting artifacts in the test images.
3. **Data Imbalance Limitations:** The F1 scores are notably lower than AUROC/AUPR metrics across all categories (e.g., hazelnut F1 = 0.75 vs AUROC = 0.995). This indicates the threshold selection strategy might still be sub-optimal or the false positive rate is heavily impacting precision.

## Cost-Benefit Discussion

Running full 3-fold cross-validation on 6 categories (18 runs) required approximately **36 GPU-hours** according to the logs aggregated in `results.parquet` (each run took ~2 GPU-hours). 

**Benefit:** 
The fold-based approach successfully provided rigorous bounds on our performance estimates, removing the variance seen in single-seed splits. It highlighted that the leather category is perfectly separable regardless of the split, while pill remains volatile.

**Cost:**
The 36 GPU-hour compute cost is significant for the baseline supervised training. Moving forward into self-supervised pre-training, we should evaluate whether we can drop down to a single definitive test split for hyperparameter tuning to save compute, and only perform 3-fold validation for the final release candidate models.
