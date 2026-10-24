# Palm Flower Classifier Model Comparison

> [!NOTE]
> **Test Split Size & Statistical Context**:
> The held-out test split is small (55 total images: 10 coconut, 33 date, 12 kithul).
> Because the coconut test slice has only 10 images, a single misclassified coconut image changes accuracy by 1.82% and class recall by 10.0%.
> Small differences of a few percent between models are therefore not statistically decisive on their own without cross-validation or a larger test set.

## 1. Overview & Efficiency

| Metric | EfficientNet-B0 | ResNet-18 | ResNet-50 |
| :--- | :---: | :---: | :---: |
| **Parameters** | 4,011,391 | 11,178,051 | 23,514,179 |
| **Weights Size** | 15.58 MB | 42.72 MB | 90.00 MB |
| **Avg CPU Latency (per img)** | 51.0 ms | 58.2 ms | 144.8 ms |
| **Test Accuracy** | **89.09%** | **92.73%** | **81.82%** |
| **Macro F1 Score** | 0.863 | 0.899 | 0.806 |

## 2. Per-Class Performance

| Model | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| EfficientNet-B0 | **coconut** | 0.692 | 0.900 | 0.783 | 10.0 |
| EfficientNet-B0 | **date** | 0.968 | 0.909 | 0.938 | 33.0 |
| EfficientNet-B0 | **kithul** | 0.909 | 0.833 | 0.870 | 12.0 |
| *EfficientNet-B0 (Macro Avg)* | *Average* | *0.856* | *0.881* | *0.863* | *55.0* |
| ResNet-18 | **coconut** | 0.750 | 0.900 | 0.818 | 10.0 |
| ResNet-18 | **date** | 0.970 | 0.970 | 0.970 | 33.0 |
| ResNet-18 | **kithul** | 1.000 | 0.833 | 0.909 | 12.0 |
| *ResNet-18 (Macro Avg)* | *Average* | *0.907* | *0.901* | *0.899* | *55.0* |
| ResNet-50 | **coconut** | 0.500 | 0.900 | 0.643 | 10.0 |
| ResNet-50 | **date** | 0.963 | 0.788 | 0.867 | 33.0 |
| ResNet-50 | **kithul** | 1.000 | 0.833 | 0.909 | 12.0 |
| *ResNet-50 (Macro Avg)* | *Average* | *0.821* | *0.840* | *0.806* | *55.0* |

## 3. Confusion Matrices

### EfficientNet-B0
```
Labels: ['coconut', 'date', 'kithul']
[[ 9  1  0]
 [ 2 30  1]
 [ 2  0 10]]
```
Saved plot: `ml/results/cm_efficientnet_b0.png`

### ResNet-18
```
Labels: ['coconut', 'date', 'kithul']
[[ 9  1  0]
 [ 1 32  0]
 [ 2  0 10]]
```
Saved plot: `ml/results/cm_resnet18.png`

### ResNet-50
```
Labels: ['coconut', 'date', 'kithul']
[[ 9  1  0]
 [ 7 26  0]
 [ 2  0 10]]
```
Saved plot: `ml/results/cm_resnet50.png`

## 4. Rejection Quality (Real vs. Junk Images)

| Metric | EfficientNet-B0 | ResNet-18 | ResNet-50 |
| :--- | :---: | :---: | :---: |
| **AUROC (Max Logit)** | **0.8722** | **0.8915** | **0.7240** |
| **Real Max Logit (Mean ± Std)** | 1.32 ± 0.63 | 0.99 ± 0.45 | 0.59 ± 0.41 |
| **Junk Max Logit (Mean ± Std)** | 0.51 ± 0.28 | 0.29 ± 0.29 | 0.29 ± 0.21 |
| **Threshold at 95% Real Retention ($\tau_{95}$)** | 0.229 | 0.125 | -0.014 |
| **Junk Wrongly Accepted at $\tau_{95}$** | **28 / 33** (84.9%) | **22 / 33** (66.7%) | **32 / 33** (97.0%) |
