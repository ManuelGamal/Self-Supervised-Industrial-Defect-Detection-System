# Engineer 5 Evaluation Report  
## Supervised Industrial Defect Detection on MVTec AD

---

## 1. Project Context

This project evaluates an industrial defect detection system using images from the MVTec AD dataset. The main goal is simple: given an image of an industrial product, the model should decide whether the product is **normal** or **defective**.

The evaluation focuses on six MVTec AD categories:

- bottle
- capsule
- carpet
- hazelnut
- leather
- pill

Although the repository title mentions self-supervised industrial defect detection, the completed experimental scope for this stage is a **supervised ResNet-50 binary classification baseline**. This means the trained models use labeled examples of normal and defective images.

Engineer 5 is responsible for the evaluation side of the project. This includes computing metrics, adding confidence intervals, evaluating trained checkpoints, generating diagnostic plots, supporting qualitative analysis, writing the evaluation notebook, and preparing the final report.

---

## 2. Evaluation Scope

The updated final evaluation tests all trained fold checkpoints, not just one selected checkpoint per category.

The evaluation covers:

- 6 categories
- 3 folds per category
- 18 total checkpoint evaluations

In other words:

6 categories × 3 folds = 18 checkpoint evaluations

The classification task is binary:

- class 0 = normal
- class 1 = defective

The evaluation metrics are:

- AUROC
- AUPR
- F1 score
- Accuracy
- 95% bootstrap confidence intervals saved in the per-fold metric outputs

The diagnostic and qualitative outputs include:

- ROC curves
- Precision-Recall curves
- Confusion matrices
- Optional Grad-CAM galleries
- Optional failure-case galleries

The final notebook used for this evaluation is:

`notebooks/04_Evaluation.ipynb`

The notebook was run on Kaggle using:

- `manuelgamal/mvtec-subset` for image data
- `ssidds-checkpoints` for the trained model checkpoints
- the GitHub repository for evaluation code and split CSVs

---

## 3. Model and Training Background

### 3.1 Model Architecture

The evaluated model is a supervised binary classifier based on a ResNet-50 backbone.

The pipeline is:

Input image → ResNet-50 encoder → Global Average Pooling → Dropout → Linear classification head

The model outputs two logits:

- logit 0: normal
- logit 1: defective

During evaluation, the defect probability is computed using softmax:

defect probability = softmax(logits)[class 1]

This probability score is then used to compute AUROC, AUPR, F1, and accuracy.

---

### 3.2 Training Setup

The models were trained using:

- ResNet-50 feature extractor
- binary classification head
- focal loss
- AdamW optimizer
- cosine learning-rate schedule
- PyTorch Lightning

Engineer 3 completed the full supervised training pipeline and executed all 18 training runs.

The locked training scope was:

100% labels only × 3 folds × 6 categories

The original plan included label-ratio experiments at 10%, 50%, and 100%, but the final project scope changed. Instead of testing different label fractions, the project focused on 3-fold cross-validation for stability analysis.

---

## 4. Threshold Protocol

A very important update in this evaluation is the threshold protocol.

The previous evaluation used thresholds chosen from the evaluation predictions. That can make F1 and accuracy look slightly optimistic because the threshold is optimized on the same data being reported.

The updated notebook uses **Engineer 3 fixed category thresholds** instead. These thresholds were selected using F1-optimal search on the validation set during training.

| Category | Engineer 3 Threshold |
|---|---:|
| bottle | 0.3924 |
| capsule | 0.4337 |
| carpet | 0.4542 |
| hazelnut | 0.3982 |
| leather | 0.4731 |
| pill | 0.4564 |

The same threshold is used for all three folds of the same category.

For example:

- bottle fold 1 uses threshold 0.3924
- bottle fold 2 uses threshold 0.3924
- bottle fold 3 uses threshold 0.3924

This is because Engineer 3 reported one category-level threshold, not a separate threshold for every fold.

This makes the updated evaluation more realistic because it does not tune the threshold on the test predictions.

---

## 5. Evaluation Implementation

The evaluation code is organized under:

`src/evaluation/`

The main files are:

- `src/evaluation/metrics.py`
- `src/evaluation/bootstrap.py`
- `src/evaluation/evaluator.py`
- `src/evaluation/qualitative.py`

---

### 5.1 Metrics

The file `src/evaluation/metrics.py` implements:

- `compute_auroc`
- `compute_aupr`
- `compute_f1`
- `compute_f1_optimal`
- `compute_accuracy`
- `compute_pixel_iou`
- `evaluate_detector`

The metrics file also includes strict input validation. It checks for:

- empty arrays
- shape mismatch between labels and scores
- single-class labels when the metric requires both classes

This prevents silent evaluation mistakes and gives clearer errors when the input is invalid.

---

### 5.2 Bootstrap Confidence Intervals

The file `src/evaluation/bootstrap.py` implements bootstrap confidence intervals.

The evaluation uses:

- 10,000 bootstrap resamples
- 95% confidence intervals
- seed = 42

Bootstrap confidence intervals are important because some test sets are small. A single metric value can look strong or weak by chance, so confidence intervals help show how stable the estimate is.

The per-fold `metrics.json` files store the detailed metric outputs, including confidence interval values.

---

### 5.3 Checkpoint Evaluation

For each category and fold, the notebook:

1. Loads the fold checkpoint.
2. Builds the test dataloader.
3. Runs inference.
4. Gets the defect probability.
5. Applies the fixed Engineer 3 threshold.
6. Computes AUROC, AUPR, F1, and accuracy.
7. Computes bootstrap confidence intervals.
8. Saves outputs for that category/fold.

For every category/fold, the notebook saves:

- `metrics.json`
- `predictions.csv`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`

The output directory is:

`/kaggle/working/eval_results_all_folds/`

---

### 5.4 Qualitative Evaluation

The file `src/evaluation/qualitative.py` supports:

- Grad-CAM galleries
- failure-case galleries

In the updated all-fold notebook, qualitative generation is optional and disabled by default:

`RUN_QUALITATIVE = False`

This keeps the all-fold quantitative evaluation faster and more stable. Grad-CAM and failure-case galleries can still be generated by setting:

`RUN_QUALITATIVE = True`

---

## 6. Final Quantitative Results

The updated evaluation successfully evaluated all 18 fold checkpoints.

Successful folds: 18 / 18  
Errors: 0

This means every category and every fold was evaluated successfully.

---

### 6.1 Per-Fold Results

| Category | Fold | Checkpoint Folder | Threshold | AUROC | AUPR | F1 | Accuracy | Samples | Positive | Negative |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 1 | fold_0 | 0.3924 | 0.9841 | 0.9603 | 0.8750 | 0.9545 | 44 | 9 | 35 |
| bottle | 2 | fold_1 | 0.3924 | 0.9270 | 0.8397 | 0.8000 | 0.9318 | 44 | 9 | 35 |
| bottle | 3 | fold_2 | 0.3924 | 0.9746 | 0.9094 | 0.7500 | 0.9091 | 44 | 9 | 35 |
| capsule | 1 | fold_0 | 0.4337 | 0.8834 | 0.8093 | 0.6452 | 0.7925 | 53 | 16 | 37 |
| capsule | 2 | fold_1 | 0.4337 | 0.9375 | 0.9037 | 0.8000 | 0.8868 | 53 | 16 | 37 |
| capsule | 3 | fold_2 | 0.4337 | 0.9409 | 0.9321 | 0.8485 | 0.9057 | 53 | 16 | 37 |
| carpet | 1 | fold_0 | 0.4542 | 0.9574 | 0.8726 | 0.7742 | 0.8833 | 60 | 13 | 47 |
| carpet | 2 | fold_1 | 0.4542 | 0.9345 | 0.6666 | 0.7333 | 0.8667 | 60 | 13 | 47 |
| carpet | 3 | fold_2 | 0.4542 | 0.8854 | 0.7364 | 0.6429 | 0.8333 | 60 | 13 | 47 |
| hazelnut | 1 | fold_0 | 0.3982 | 0.9846 | 0.9330 | 0.8000 | 0.9474 | 76 | 11 | 65 |
| hazelnut | 2 | fold_1 | 0.3982 | 0.9580 | 0.8948 | 0.9000 | 0.9737 | 76 | 11 | 65 |
| hazelnut | 3 | fold_2 | 0.3982 | 0.9916 | 0.9633 | 0.8571 | 0.9605 | 76 | 11 | 65 |
| leather | 1 | fold_0 | 0.4731 | 0.9949 | 0.9860 | 0.8889 | 0.9464 | 56 | 14 | 42 |
| leather | 2 | fold_1 | 0.4731 | 1.0000 | 1.0000 | 0.9630 | 0.9821 | 56 | 14 | 42 |
| leather | 3 | fold_2 | 0.4731 | 0.9966 | 0.9911 | 0.9286 | 0.9643 | 56 | 14 | 42 |
| pill | 1 | fold_0 | 0.4564 | 0.8946 | 0.8760 | 0.7660 | 0.8333 | 66 | 22 | 44 |
| pill | 2 | fold_1 | 0.4564 | 0.9029 | 0.8494 | 0.6977 | 0.8030 | 66 | 22 | 44 |
| pill | 3 | fold_2 | 0.4564 | 0.9360 | 0.9048 | 0.7917 | 0.8485 | 66 | 22 | 44 |

---

### 6.2 Mean Results Across Folds

| Category | Threshold | Mean AUROC | AUROC Std | Mean AUPR | AUPR Std | Mean F1 | F1 Std | Mean Accuracy | Accuracy Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 0.3924 | 0.9619 | 0.0306 | 0.9031 | 0.0606 | 0.8083 | 0.0629 | 0.9318 | 0.0227 |
| capsule | 0.4337 | 0.9206 | 0.0322 | 0.8817 | 0.0643 | 0.7645 | 0.1062 | 0.8616 | 0.0607 |
| carpet | 0.4542 | 0.9258 | 0.0368 | 0.7585 | 0.1048 | 0.7168 | 0.0672 | 0.8611 | 0.0255 |
| hazelnut | 0.3982 | 0.9781 | 0.0177 | 0.9304 | 0.0343 | 0.8524 | 0.0502 | 0.9605 | 0.0132 |
| leather | 0.4731 | 0.9972 | 0.0026 | 0.9923 | 0.0071 | 0.9268 | 0.0371 | 0.9643 | 0.0179 |
| pill | 0.4564 | 0.9112 | 0.0219 | 0.8767 | 0.0277 | 0.7518 | 0.0486 | 0.8283 | 0.0231 |
| **Overall Mean** | **0.4347** | **0.9491** | **0.0350** | **0.8905** | **0.0772** | **0.8034** | **0.0766** | **0.9013** | **0.0582** |

Total unique test samples across categories:

355 samples

The same category-level test split is used for all three fold checkpoints of a category, while the checkpoint changes across folds. This allows the three trained fold models to be compared directly on the same held-out test set.

---

## 7. Per-Category Discussion

### 7.1 Leather

Leather is the strongest category overall.

It achieved:

- Mean AUROC = 0.9972
- Mean AUPR = 0.9923
- Mean F1 = 0.9268
- Mean Accuracy = 0.9643

Leather also has the lowest AUROC standard deviation across folds:

Leather AUROC Std = 0.0026

This means the model performs very consistently on leather. The defects are likely visually clear enough for the model to learn stable features across folds.

---

### 7.2 Hazelnut

Hazelnut also performs strongly.

It achieved:

- Mean AUROC = 0.9781
- Mean AUPR = 0.9304
- Mean F1 = 0.8524
- Mean Accuracy = 0.9605

The model is reliable on hazelnut, although F1 varies slightly across folds. This means the ranking performance is very strong, while the final threshold-based decision still changes a bit from fold to fold.

---

### 7.3 Bottle

Bottle achieved:

- Mean AUROC = 0.9619
- Mean AUPR = 0.9031
- Mean F1 = 0.8083
- Mean Accuracy = 0.9318

Bottle has strong overall performance, but fold 2 is weaker than fold 1 and fold 3. This shows why it is useful to evaluate all folds instead of only reporting a single best model.

---

### 7.4 Capsule

Capsule achieved:

- Mean AUROC = 0.9206
- Mean AUPR = 0.8817
- Mean F1 = 0.7645
- Mean Accuracy = 0.8616

Capsule has the highest F1 standard deviation:

Capsule F1 Std = 0.1062

This suggests that capsule classification is sensitive to the training fold. The defects may be subtle, small, or visually close to normal capsule variations.

---

### 7.5 Carpet

Carpet is one of the hardest categories.

It achieved:

- Mean AUROC = 0.9258
- Mean AUPR = 0.7585
- Mean F1 = 0.7168
- Mean Accuracy = 0.8611

Carpet has the lowest mean F1 and lowest mean AUPR. This suggests that threshold-based classification is difficult for this category. Repeated texture patterns and subtle defects can make carpet images harder to classify correctly.

---

### 7.6 Pill

Pill has the lowest mean AUROC.

It achieved:

- Mean AUROC = 0.9112
- Mean AUPR = 0.8767
- Mean F1 = 0.7518
- Mean Accuracy = 0.8283

Pill defects can include small cracks, discoloration, or shape irregularities. These defects may not dominate the full image, which makes image-level classification more challenging.

---

## 8. Fold Consistency Discussion

The all-fold evaluation gives a clearer picture than a best-checkpoint-only evaluation.

Stable categories:

- leather
- hazelnut

More variable categories:

- capsule
- carpet
- bottle

Leather is the most stable category, with AUROC standard deviation of only 0.0026.

Carpet has the highest AUROC standard deviation:

Carpet AUROC Std = 0.0368

Capsule has the highest F1 standard deviation:

Capsule F1 Std = 0.1062

This shows that a single fold can hide important behavior. The 3-fold setup is useful because it shows whether the model is consistently strong or only strong on one split.

---

## 9. Negative Results and Limitations

### 9.1 Carpet has the weakest F1 and AUPR

Carpet has:

- Mean F1 = 0.7168
- Mean AUPR = 0.7585

This makes carpet one of the weakest categories in the updated evaluation. The model can still rank some defective images well, but the final threshold-based classification is harder.

---

### 9.2 Pill has the weakest AUROC and accuracy

Pill has:

- Mean AUROC = 0.9112
- Mean Accuracy = 0.8283

This indicates that pill is challenging both in ranking and in final classification. Small or subtle defects may not be captured well enough by image-level supervision alone.

---

### 9.3 Fixed thresholds reduce inflated F1 and accuracy

The updated evaluation uses Engineer 3 validation thresholds. This is more realistic than choosing a new threshold directly from test predictions.

Because the threshold is fixed, F1 and accuracy may be lower than the previous report. This is expected and is not a problem. It means the protocol is more honest and closer to deployment.

---

### 9.4 Label-ratio ablation was removed from scope

The original plan included:

- 10% labels
- 50% labels
- 100% labels

The final scope only evaluates 100% labels with 3-fold cross-validation. Therefore, the current report does not show how the model performs in low-label settings.

---

### 9.5 The evaluation is image-level, not segmentation-level

The model predicts whether the whole image is defective or normal. It does not produce a true pixel-level defect mask.

Grad-CAM can help visualize where the model is looking, but it is not the same as a segmentation model.

---

### 9.6 Same test split is reused across fold checkpoints

The updated notebook evaluates all three fold checkpoints for each category on the same category-level held-out test split.

This is useful because it makes fold checkpoints directly comparable. However, it also means the three folds are not evaluated on three independent test sets.

---

## 10. GPU-Hour Cost-Benefit Discussion

The training phase involved:

18 training runs = 6 categories × 3 folds

Assuming each Kaggle T4 run took approximately 30 minutes, the total training cost is approximately:

18 × 0.5 GPU-hours = 9 GPU-hours

The updated evaluation completed all 18 checkpoint evaluations in approximately 13 minutes.

This means the evaluation cost is small compared with the training cost.

The 3-fold setup costs more than a single split, but it gives better evidence about model stability. The fold-level results show that some categories, especially capsule and carpet, vary across folds. Therefore, the additional compute cost is justified.

---

## 11. Test Coverage and Code Quality

Engineer 5 added and updated tests for:

- `tests/test_metrics.py`
- `tests/test_bootstrap.py`
- `tests/test_engineer5_coverage.py`

The final local coverage result for the main evaluation files was:

| File | Coverage |
|---|---:|
| `src/evaluation/bootstrap.py` | 92% |
| `src/evaluation/evaluator.py` | 94% |
| `src/evaluation/metrics.py` | 95% |
| `src/evaluation/qualitative.py` | 90% |
| **Total** | **93%** |

This satisfies the required coverage threshold of more than 85%.

---

## 12. Final Deliverables

Engineer 5 deliverables completed:

| Deliverable | Status |
|---|---|
| `src/evaluation/metrics.py` | Completed |
| `src/evaluation/bootstrap.py` | Completed |
| `src/evaluation/evaluator.py` | Completed |
| `src/evaluation/qualitative.py` | Completed |
| `tests/test_metrics.py` | Completed |
| `tests/test_bootstrap.py` | Completed |
| `tests/test_engineer5_coverage.py` | Completed |
| `notebooks/04_Evaluation.ipynb` | Completed |
| `docs/REPORT.md` | Completed |

---

## 13. Conclusion

The updated all-fold evaluation confirms that the supervised ResNet-50 defect classifier performs strongly overall, with an overall mean AUROC of 0.9491 across the six categories.

Leather is the strongest and most stable category. Hazelnut and bottle also perform well. Carpet and pill are the most challenging categories, with carpet having the weakest F1 and AUPR, and pill having the weakest AUROC and accuracy.

This final evaluation is stronger than the previous version because it reports all 18 fold checkpoints and uses fixed validation thresholds from Engineer 3 instead of optimizing thresholds on the test set.

---

## References

1. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

2. Bergmann P, Fauser M, Sattlegger D, Steger C. MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

3. Lin TY, Goyal P, Girshick R, He K, Dollár P. Focal Loss for Dense Object Detection. Proceedings of the IEEE International Conference on Computer Vision. 2017.

4. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Proceedings of the IEEE International Conference on Computer Vision. 2017.

5. Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall/CRC. 1993.