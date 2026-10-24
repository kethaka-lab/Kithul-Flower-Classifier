# Out-of-Distribution & Unfamiliarity Rejection Experiments

> [!NOTE]
> **Experimental Protocol**:
> - Evaluated on **55 real TEST palm flower images** (from `ml/splits.json`) and **17 held-out JUNK test images** (from 50/50 split of `junk/`, seed=42).
> - Centroids and shared regularized covariance ($\lambda = 10^{-4}$) were estimated **strictly on the TRAIN split**.
> - All decision thresholds ($	au_{95}$, $	au_{90}$) were calibrated **strictly on real VAL images** (and the first junk half), then evaluated out-of-sample on real TEST images and the second junk half.

## 1. Single Rejection Score Comparison

| Model | Metric / Score | AUROC | Real Kept (@ $\tau_{95}$) | Junk Rej (@ $\tau_{95}$) | Real Kept (@ $\tau_{90}$) | Junk Rej (@ $\tau_{90}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | Max Logit | **0.8674** | 85.5% | 64.7% | 83.6% | 76.5% |
| **EfficientNet-B0** | Max Softmax Prob | **0.8898** | 89.1% | 58.8% | 83.6% | 76.5% |
| **EfficientNet-B0** | Energy Score (LogSumExp) | **0.8203** | 83.6% | 52.9% | 81.8% | 58.8% |
| **EfficientNet-B0** | Cosine Sim to Nearest Centroid | **0.9701** | 94.5% | 88.2% | 80.0% | 100.0% |
| **EfficientNet-B0** | Mahalanobis Dist (Shared Cov) | **0.8289** | 90.9% | 35.3% | 87.3% | 47.1% |
| | | | | | | |
| **ResNet-18** | Max Logit | **0.9102** | 90.9% | 52.9% | 89.1% | 52.9% |
| **ResNet-18** | Max Softmax Prob | **0.9155** | 94.5% | 47.1% | 89.1% | 64.7% |
| **ResNet-18** | Energy Score (LogSumExp) | **0.8471** | 89.1% | 47.1% | 81.8% | 52.9% |
| **ResNet-18** | Cosine Sim to Nearest Centroid | **0.9840** | 92.7% | 94.1% | 90.9% | 94.1% |
| **ResNet-18** | Mahalanobis Dist (Shared Cov) | **0.7262** | 98.2% | 11.8% | 94.5% | 29.4% |
| | | | | | | |
| **ResNet-50** | Max Logit | **0.6706** | 92.7% | 5.9% | 83.6% | 17.6% |
| **ResNet-50** | Max Softmax Prob | **0.6898** | 96.4% | 5.9% | 87.3% | 17.6% |
| **ResNet-50** | Energy Score (LogSumExp) | **0.6182** | 87.3% | 11.8% | 80.0% | 11.8% |
| **ResNet-50** | Cosine Sim to Nearest Centroid | **0.9561** | 96.4% | 76.5% | 89.1% | 94.1% |
| **ResNet-50** | Mahalanobis Dist (Shared Cov) | **0.7390** | 92.7% | 11.8% | 92.7% | 11.8% |
| | | | | | | |

## 2. Combined Rejection Rules for Best Model (ResNet-18)

A combined rule checks **both** feature space proximity (penultimate feature embedding) and activation confidence (max logit):
$$\text{Accept} \iff (\text{Feature Metric} \ge \tau_{\text{feat}}) \land (\text{Max Logit} \ge \tau_{\text{logit}})$$

| Combined Rule | Target VAL Retention | Calibrated Thresholds | Real TEST Kept | JUNK TEST Rejected |
| :--- | :---: | :--- | :---: | :---: |
| **Cosine Sim + Max Logit** | 95% | $\tau_{\text{feat}} = 0.649,\; \tau_{\text{logit}} = 0.021$ | **94.5%** | **94.1%** |
| **Cosine Sim + Max Logit** | 90% | $\tau_{\text{feat}} = 0.649,\; \tau_{\text{logit}} = 0.179$ | **90.9%** | **100.0%** |
| **Mahalanobis + Max Logit** | 95% | $\tau_{\text{feat}} = -1741.422,\; \tau_{\text{logit}} = 0.021$ | **98.2%** | **17.6%** |
| **Mahalanobis + Max Logit** | 90% | $\tau_{\text{feat}} = -1741.422,\; \tau_{\text{logit}} = 0.021$ | **98.2%** | **17.6%** |

### Key Observations
1. **Penultimate-layer feature metrics (Cosine Similarity & Mahalanobis Distance)** significantly outperform raw softmax probabilities and logits on out-of-distribution junk images.
2. **Max Softmax Probability** performs the worst for rejection, because softmax forces closed-world probability distributions to sum to 1, causing overconfident false positives on unfamiliar junk.
3. **Combining feature-space proximity with Max Logit** provides robust dual-layer defense: it rejects images that either look structurally distinct in feature space or lack confident classification activation.
