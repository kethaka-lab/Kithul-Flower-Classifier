"""
Build rejection artifacts for palm flower classifiers:
Saves backend/rejection/<model_name>.json with:
- classes
- centroids (3 class centroids from TRAIN split)
- tau_feat_95 (threshold at 95% real-VAL retention)
- tau_feat_90 (threshold at 90% real-VAL retention)
- default tau_feat (tau_feat_95)

Also prints the test retention and junk rejection metrics to verify matching rejection_scores.md.
"""
import json
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from ml.rejection_utils import (
    MODEL_CONFIGS,
    IMAGE_EXTENSIONS,
    get_eval_transform,
    load_model,
    extract_features_and_logits,
    compute_class_centroids,
    compute_cosine_similarity,
)


def main():
    splits_path = Path("ml/splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    classes = splits["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_paths = [Path(p) for p in splits["train"]]
    val_paths = [Path(p) for p in splits["val"]]
    test_paths = [Path(p) for p in splits["test"]]

    train_labels = np.array([class_to_idx[Path(p).parent.name] for p in train_paths])
    val_labels = np.array([class_to_idx[Path(p).parent.name] for p in val_paths])
    test_labels = np.array([class_to_idx[Path(p).parent.name] for p in test_paths])

    # 50/50 split of junk images with fixed seed 42
    junk_dir = Path("junk")
    junk_paths = sorted([f for f in junk_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS])
    rng = random.Random(42)
    junk_shuffled = list(junk_paths)
    rng.shuffle(junk_shuffled)
    n_half = len(junk_shuffled) // 2
    junk_val = junk_shuffled[:n_half]
    junk_test = junk_shuffled[n_half:]

    eval_tf = get_eval_transform(224)

    out_dir = Path("backend/rejection")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BUILDING REJECTION ARTIFACTS (Cosine Centroid Rejection)")
    print("=" * 65)

    for cfg in MODEL_CONFIGS:
        mid = cfg["id"]
        w_path = Path(cfg["weights_path"])
        if not w_path.exists():
            print(f"Skipping {mid}: weights not found at {w_path}")
            continue

        model, ckpt = load_model(cfg["timm_name"], w_path, len(classes))

        train_feats, _ = extract_features_and_logits(model, train_paths, eval_tf)
        val_feats, _ = extract_features_and_logits(model, val_paths, eval_tf)
        test_feats, _ = extract_features_and_logits(model, test_paths, eval_tf)
        junk_test_feats, _ = extract_features_and_logits(model, junk_test, eval_tf)

        # Compute centroids on TRAIN features only
        centroids = compute_class_centroids(train_feats, train_labels, len(classes))

        # Compute cosine similarity
        val_cos = compute_cosine_similarity(val_feats, centroids)
        test_cos = compute_cosine_similarity(test_feats, centroids)
        junk_test_cos = compute_cosine_similarity(junk_test_feats, centroids)

        # Calibrate thresholds on VAL
        tau_95 = float(np.percentile(val_cos, 5.0))
        tau_90 = float(np.percentile(val_cos, 10.0))

        real_kept_95 = float((test_cos >= tau_95).mean())
        junk_rej_95 = float((junk_test_cos < tau_95).mean())

        real_kept_90 = float((test_cos >= tau_90).mean())
        junk_rej_90 = float((junk_test_cos < tau_90).mean())

        artifact_data = {
            "model_name": mid,
            "classes": classes,
            "tau_feat": round(tau_95, 4),
            "tau_feat_95": round(tau_95, 4),
            "tau_feat_90": round(tau_90, 4),
            "metric": "cosine_similarity_nearest_centroid",
            "centroids": centroids.tolist(),
            "validation_eval": {
                "val_samples": len(val_paths),
                "val_kept_at_tau95": float((val_cos >= tau_95).mean()),
                "val_kept_at_tau90": float((val_cos >= tau_90).mean()),
            },
            "test_eval": {
                "test_samples": len(test_paths),
                "junk_test_samples": len(junk_test),
                "real_kept_95": round(real_kept_95, 4),
                "junk_rej_95": round(junk_rej_95, 4),
                "real_kept_90": round(real_kept_90, 4),
                "junk_rej_90": round(junk_rej_90, 4),
            },
        }

        artifact_path = out_dir / f"{mid}.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2)

        print(f"\nModel: {cfg['display_name']} -> saved {artifact_path}")
        print(f"  tau_feat (95% VAL retention) = {tau_95:.4f}")
        print(f"    Real TEST kept: {real_kept_95*100:.1f}% | JUNK TEST rejected: {junk_rej_95*100:.1f}%")
        print(f"  tau_feat (90% VAL retention) = {tau_90:.4f}")
        print(f"    Real TEST kept: {real_kept_90*100:.1f}% | JUNK TEST rejected: {junk_rej_90*100:.1f}%")

    print("\nRejection artifacts saved successfully.")


if __name__ == "__main__":
    main()
