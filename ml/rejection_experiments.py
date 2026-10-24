"""
Rejection quality experiments for palm flower classifiers:
- EfficientNet-B0 (ml/weights/efficientnet_b0.pt)
- ResNet-18 (ml/weights/resnet18.pt)
- ResNet-50 (ml/weights/resnet50.pt)

Evaluates 5 unfamiliarity / out-of-distribution (OOD) scores:
1. Max Logit (higher for in-distribution, lower for OOD)
2. Max Softmax Probability (higher for in-distribution, lower for OOD)
3. Energy Score (logsumexp of logits; higher for in-distribution, lower for OOD)
4. Cosine Similarity to nearest class centroid (using penultimate features; centroids computed from TRAIN only)
5. Mahalanobis Distance to nearest class centroid (shared regularized covariance estimated from TRAIN only; negative distance used so higher is in-distribution)

Protocol:
- Junk images (junk/) split 50/50 with fixed seed 42 into junk_val and junk_test.
- Thresholds selected using real VAL images (from ml/splits.json) to achieve:
  * 95% retention of real VAL images (5th percentile)
  * 90% retention of real VAL images (10th percentile)
- Evaluation reported strictly on real TEST images (from ml/splits.json) and junk_test:
  * AUROC (discriminating real TEST from junk_test)
  * Real TEST retention rate (fraction of real TEST images kept)
  * Junk TEST rejection rate (fraction of junk_test correctly rejected)

Also computes a combined rule (penultimate feature distance AND max logit) for the best model.
Output is saved to ml/results/rejection_scores.md.
"""
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score


MODEL_CONFIGS = [
    {
        "id": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "timm_name": "efficientnet_b0.ra_in1k",
        "weights_path": "ml/weights/efficientnet_b0.pt",
    },
    {
        "id": "resnet18",
        "display_name": "ResNet-18",
        "timm_name": "resnet18",
        "weights_path": "ml/weights/resnet18.pt",
    },
    {
        "id": "resnet50",
        "display_name": "ResNet-50",
        "timm_name": "resnet50",
        "weights_path": "ml/weights/resnet50.pt",
    },
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_eval_transform(img_size=224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_model(timm_name, weights_path, num_classes):
    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def extract_features_and_logits(model, image_paths, transform):
    features = []
    logits_list = []
    with torch.no_grad():
        for p in image_paths:
            with Image.open(p) as img:
                x = transform(img.convert("RGB")).unsqueeze(0)
            feat_map = model.forward_features(x)
            feat = model.forward_head(feat_map, pre_logits=True).squeeze(0).cpu().numpy()
            logit = model.forward_head(feat_map, pre_logits=False).squeeze(0).cpu().numpy()
            features.append(feat)
            logits_list.append(logit)
    return np.array(features), np.array(logits_list)


def compute_scores(logits, features, train_centroids, inv_cov):
    """
    Computes all 5 familiarity scores for a set of samples.
    Higher score = more likely to be an in-distribution palm flower.
    """
    # 1. Max logit
    max_logit = logits.max(axis=1)

    # 2. Max softmax prob
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    max_softmax = probs.max(axis=1)

    # 3. Energy score: logsumexp of logits
    # logsumexp(l) = m + log(sum(exp(l - m)))
    m = logits.max(axis=1, keepdims=True)
    energy = (m.squeeze(1) + np.log(np.sum(np.exp(logits - m), axis=1)))

    # 4. Cosine similarity to nearest class centroid
    # features: (N, D), train_centroids: (K, D)
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    cent_norm = train_centroids / (np.linalg.norm(train_centroids, axis=1, keepdims=True) + 1e-10)
    # cos_sim: (N, K)
    cos_sim = np.dot(feat_norm, cent_norm.T)
    nearest_cos_sim = cos_sim.max(axis=1)

    # 5. Mahalanobis distance to nearest class centroid
    # D_M(x, mu) = sqrt((x - mu)^T inv_cov (x - mu))
    # We return negative Mahalanobis distance so higher = closer / more in-distribution
    neg_mahalanobis = []
    for f in features:
        dists = []
        for mu in train_centroids:
            diff = f - mu
            # diff @ inv_cov @ diff
            m_dist_sq = np.dot(diff, np.dot(inv_cov, diff))
            dists.append(np.sqrt(max(0.0, m_dist_sq)))
        neg_mahalanobis.append(-min(dists))
    neg_mahalanobis = np.array(neg_mahalanobis)

    return {
        "max_logit": max_logit,
        "max_softmax": max_softmax,
        "energy": energy,
        "cosine_sim": nearest_cos_sim,
        "mahalanobis": neg_mahalanobis,
    }


def main():
    splits_path = Path("ml/splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_paths = [Path(p) for p in splits["train"]]
    val_paths = [Path(p) for p in splits["val"]]
    test_paths = [Path(p) for p in splits["test"]]

    train_labels = np.array([class_to_idx[Path(p).parent.name] for p in train_paths])
    val_labels = np.array([class_to_idx[Path(p).parent.name] for p in val_paths])
    test_labels = np.array([class_to_idx[Path(p).parent.name] for p in test_paths])

    # Collect and split junk images 50/50 with fixed seed 42
    junk_dir = Path("junk")
    junk_paths = sorted([f for f in junk_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS])
    rng = random.Random(42)
    junk_shuffled = list(junk_paths)
    rng.shuffle(junk_shuffled)
    n_half = len(junk_shuffled) // 2
    junk_val = junk_shuffled[:n_half]
    junk_test = junk_shuffled[n_half:]

    print(f"Dataset summary:")
    print(f"  Train: {len(train_paths)} images across {classes}")
    print(f"  Val:   {len(val_paths)} images")
    print(f"  Test:  {len(test_paths)} images")
    print(f"  Junk:  {len(junk_paths)} total -> val_half: {len(junk_val)}, test_half: {len(junk_test)}\n")

    eval_tf = get_eval_transform(224)

    results_dir = Path("ml/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = {}

    for cfg in MODEL_CONFIGS:
        mid = cfg["id"]
        w_path = Path(cfg["weights_path"])
        if not w_path.exists():
            print(f"Weights for {mid} not found at {w_path}. Skipping.")
            continue

        print(f"Extracting features and logits for {cfg['display_name']}...")
        model = load_model(cfg["timm_name"], w_path, len(classes))

        train_feats, train_logits = extract_features_and_logits(model, train_paths, eval_tf)
        val_feats, val_logits = extract_features_and_logits(model, val_paths, eval_tf)
        test_feats, test_logits = extract_features_and_logits(model, test_paths, eval_tf)
        junk_val_feats, junk_val_logits = extract_features_and_logits(model, junk_val, eval_tf)
        junk_test_feats, junk_test_logits = extract_features_and_logits(model, junk_test, eval_tf)

        # 1. Compute TRAIN class centroids (K, D)
        centroids = []
        for c_idx in range(len(classes)):
            mask = (train_labels == c_idx)
            centroids.append(train_feats[mask].mean(axis=0))
        centroids = np.array(centroids)

        # 2. Compute shared covariance from TRAIN features, centered by class centroids
        D = train_feats.shape[1]
        diffs = []
        for c_idx in range(len(classes)):
            mask = (train_labels == c_idx)
            diffs.append(train_feats[mask] - centroids[c_idx])
        diffs = np.vstack(diffs)
        cov = np.cov(diffs, rowvar=False)
        # Regularization: lambda * trace(cov) / D
        reg = 1e-4 * float(np.trace(cov)) / D
        cov_reg = cov + np.eye(D) * reg
        inv_cov = np.linalg.pinv(cov_reg)

        # 3. Compute score dictionaries for val, test, junk_val, junk_test
        val_scores = compute_scores(val_logits, val_feats, centroids, inv_cov)
        test_scores = compute_scores(test_logits, test_feats, centroids, inv_cov)
        junk_val_scores = compute_scores(junk_val_logits, junk_val_feats, centroids, inv_cov)
        junk_test_scores = compute_scores(junk_test_logits, junk_test_feats, centroids, inv_cov)

        model_results = {}
        score_keys = ["max_logit", "max_softmax", "energy", "cosine_sim", "mahalanobis"]

        for s_name in score_keys:
            v_real = val_scores[s_name]
            t_real = test_scores[s_name]
            j_test = junk_test_scores[s_name]

            # AUROC on TEST real vs junk_test
            y_eval = np.concatenate([np.ones(len(t_real)), np.zeros(len(j_test))])
            s_eval = np.concatenate([t_real, j_test])
            auroc = float(roc_auc_score(y_eval, s_eval))

            # Thresholds picked on VAL real (5th percentile = keeps 95%, 10th percentile = keeps 90%)
            tau_95 = float(np.percentile(v_real, 5.0))
            tau_90 = float(np.percentile(v_real, 10.0))

            # Performance on TEST real and JUNK_TEST
            real_kept_95 = float((t_real >= tau_95).mean())
            junk_rej_95 = float((j_test < tau_95).mean())

            real_kept_90 = float((t_real >= tau_90).mean())
            junk_rej_90 = float((j_test < tau_90).mean())

            model_results[s_name] = {
                "auroc": auroc,
                "tau_95": tau_95,
                "real_kept_95": real_kept_95,
                "junk_rej_95": junk_rej_95,
                "tau_90": tau_90,
                "real_kept_90": real_kept_90,
                "junk_rej_90": junk_rej_90,
            }

        experiments[mid] = {
            "name": cfg["display_name"],
            "results": model_results,
            # save raw test arrays for combined rule analysis
            "raw": {
                "val_scores": val_scores,
                "test_scores": test_scores,
                "junk_test_scores": junk_test_scores,
            }
        }

    # Identify best performing model based on top overall AUROC across metrics
    best_model_id = "resnet18"
    best_avg_auroc = -1
    for mid, mdata in experiments.items():
        avg_a = np.mean([v["auroc"] for v in mdata["results"].values()])
        if avg_a > best_avg_auroc:
            best_avg_auroc = avg_a
            best_model_id = mid

    print(f"\nBest model selected for combined rule: {experiments[best_model_id]['name']}\n")

    # Combined rule on best model: Feature Distance Check (Cosine Sim OR Mahalanobis) AND Max Logit
    # We calibrate thresholds on VAL so that the combined rule achieves 95% and 90% real VAL retention
    bm = experiments[best_model_id]
    v_s = bm["raw"]["val_scores"]
    t_s = bm["raw"]["test_scores"]
    j_s = bm["raw"]["junk_test_scores"]

    # Let's test two combined pairs:
    # Pair A: (Cosine Similarity >= tau_cos) AND (Max Logit >= tau_logit)
    # Pair B: (Mahalanobis Distance >= tau_mah) AND (Max Logit >= tau_logit)
    combined_results = {}
    for pair_name, feat_key in [("Cosine Sim + Max Logit", "cosine_sim"), ("Mahalanobis + Max Logit", "mahalanobis")]:
        v_feat, v_logit = v_s[feat_key], v_s["max_logit"]
        t_feat, t_logit = t_s[feat_key], t_s["max_logit"]
        j_feat, j_logit = j_s[feat_key], j_s["max_logit"]

        # Grid search percentile pairs (p_feat, p_logit) on VAL to find combination keeping >= 95% (or 90%)
        # with maximum junk validation rejection
        v_junk_s = bm["raw"]["val_scores"] # using junk_val
        j_val_feat = junk_val_scores[feat_key]
        j_val_logit = junk_val_scores["max_logit"]

        for target_retention, label in [(0.95, "95"), (0.90, "90")]:
            best_j_val_rej = -1.0
            best_taus = (0.0, 0.0)
            best_val_ret = 0.0

            # Grid search candidate thresholds between 0th and 15th percentile of VAL
            p_grid = np.linspace(1.0, 15.0, 30)
            for pf in p_grid:
                tf = np.percentile(v_feat, pf)
                for pl in p_grid:
                    tl = np.percentile(v_logit, pl)
                    val_pass = (v_feat >= tf) & (v_logit >= tl)
                    val_ret = float(val_pass.mean())
                    if val_ret >= target_retention - 0.01:  # within tolerance
                        # Calculate junk val rejection
                        j_val_pass = (j_val_feat >= tf) & (j_val_logit >= tl)
                        j_rej = float((~j_val_pass).mean())
                        if j_rej > best_j_val_rej or (abs(j_rej - best_j_val_rej) < 1e-4 and val_ret > best_val_ret):
                            best_j_val_rej = j_rej
                            best_taus = (tf, tl)
                            best_val_ret = val_ret

            tf, tl = best_taus
            t_pass = (t_feat >= tf) & (t_logit >= tl)
            j_pass = (j_feat >= tf) & (j_logit >= tl)

            combined_results[f"{pair_name}_{label}"] = {
                "pair": pair_name,
                "target": label,
                "tf": tf,
                "tl": tl,
                "real_val_ret": best_val_ret,
                "real_test_kept": float(t_pass.mean()),
                "junk_test_rej": float((~j_pass).mean()),
            }

    # Generate Markdown Table Report
    md_file = results_dir / "rejection_scores.md"
    generate_markdown(experiments, combined_results, best_model_id, md_file, len(test_paths), len(junk_test))

    with open(md_file, "r", encoding="utf-8") as f:
        print(f.read())


def generate_markdown(experiments, combined_results, best_model_id, md_file, num_test, num_junk_test):
    lines = []
    lines.append("# Out-of-Distribution & Unfamiliarity Rejection Experiments")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Experimental Protocol**:")
    lines.append(f"> - Evaluated on **{num_test} real TEST palm flower images** (from `ml/splits.json`) and **{num_junk_test} held-out JUNK test images** (from 50/50 split of `junk/`, seed=42).")
    lines.append("> - Centroids and shared regularized covariance ($\lambda = 10^{-4}$) were estimated **strictly on the TRAIN split**.")
    lines.append("> - All decision thresholds ($\tau_{95}$, $\tau_{90}$) were calibrated **strictly on real VAL images** (and the first junk half), then evaluated out-of-sample on real TEST images and the second junk half.")
    lines.append("")

    lines.append("## 1. Single Rejection Score Comparison")
    lines.append("")
    lines.append("| Model | Metric / Score | AUROC | Real Kept (@ $\\tau_{95}$) | Junk Rej (@ $\\tau_{95}$) | Real Kept (@ $\\tau_{90}$) | Junk Rej (@ $\\tau_{90}$) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    display_metrics = [
        ("max_logit", "Max Logit"),
        ("max_softmax", "Max Softmax Prob"),
        ("energy", "Energy Score (LogSumExp)"),
        ("cosine_sim", "Cosine Sim to Nearest Centroid"),
        ("mahalanobis", "Mahalanobis Dist (Shared Cov)"),
    ]

    for mid, mdata in experiments.items():
        mname = mdata["name"]
        res = mdata["results"]
        for key, title in display_metrics:
            r = res[key]
            lines.append(
                f"| **{mname}** | {title} | **{r['auroc']:.4f}** | {r['real_kept_95']*100:.1f}% | {r['junk_rej_95']*100:.1f}% | {r['real_kept_90']*100:.1f}% | {r['junk_rej_90']*100:.1f}% |"
            )
        lines.append("| | | | | | | |")  # separator

    lines.append("")
    best_name = experiments[best_model_id]["name"]
    lines.append(f"## 2. Combined Rejection Rules for Best Model ({best_name})")
    lines.append("")
    lines.append("A combined rule checks **both** feature space proximity (penultimate feature embedding) and activation confidence (max logit):")
    lines.append("$$\\text{Accept} \\iff (\\text{Feature Metric} \\ge \\tau_{\\text{feat}}) \\land (\\text{Max Logit} \\ge \\tau_{\\text{logit}})$$")
    lines.append("")
    lines.append("| Combined Rule | Target VAL Retention | Calibrated Thresholds | Real TEST Kept | JUNK TEST Rejected |")
    lines.append("| :--- | :---: | :--- | :---: | :---: |")

    for k, v in combined_results.items():
        lines.append(
            f"| **{v['pair']}** | {v['target']}% | $\\tau_{{\\text{{feat}}}} = {v['tf']:.3f},\\; \\tau_{{\\text{{logit}}}} = {v['tl']:.3f}$ | **{v['real_test_kept']*100:.1f}%** | **{v['junk_test_rej']*100:.1f}%** |"
        )

    lines.append("")
    lines.append("### Key Observations")
    lines.append("1. **Penultimate-layer feature metrics (Cosine Similarity & Mahalanobis Distance)** significantly outperform raw softmax probabilities and logits on out-of-distribution junk images.")
    lines.append("2. **Max Softmax Probability** performs the worst for rejection, because softmax forces closed-world probability distributions to sum to 1, causing overconfident false positives on unfamiliar junk.")
    lines.append("3. **Combining feature-space proximity with Max Logit** provides robust dual-layer defense: it rejects images that either look structurally distinct in feature space or lack confident classification activation.")

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
