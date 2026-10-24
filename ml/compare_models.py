"""
Model comparison evaluation script for palm flower classifiers:
- EfficientNet-B0 (ml/weights/efficientnet_b0.pt)
- ResNet-18 (ml/weights/resnet18.pt)
- ResNet-50 (ml/weights/resnet50.pt)

Evaluates on the shared test split from ml/splits.json, computes confusion matrices,
parameter counts, weight sizes, inference latency on CPU, and out-of-distribution
rejection capability against the junk/ folder.

Outputs:
- ml/results/comparison.json
- ml/results/comparison.md
- ml/results/cm_<model_name>.png
"""
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


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
    return model, ckpt


def save_confusion_matrix_image(cm, classes, output_path, title):
    # Try importing matplotlib if user installed it, otherwise render cleanly with Pillow
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import itertools

        fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return
    except ImportError:
        pass

    # Pure Pillow fallback for crisp image generation without extra dependencies
    n = len(classes)
    cell_w, cell_h = 100, 70
    margin_left, margin_top, margin_bottom, margin_right = 120, 80, 50, 40
    width = margin_left + n * cell_w + margin_right
    height = margin_top + n * cell_h + margin_bottom

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_body = ImageFont.truetype("arial.ttf", 13)
        font_bold = ImageFont.truetype("arialbd.ttf", 15)
    except Exception:
        font_title = font_body = font_bold = ImageFont.load_default()

    draw.text((margin_left, 25), title, fill=(20, 20, 20), font=font_title)
    draw.text((15, margin_top - 30), "True \\ Pred", fill=(70, 70, 70), font=font_body)

    for j, c in enumerate(classes):
        tx = margin_left + j * cell_w + cell_w // 2 - 20
        draw.text((tx, margin_top - 25), c, fill=(30, 30, 30), font=font_bold)

    for i, c in enumerate(classes):
        ty = margin_top + i * cell_h + cell_h // 2 - 8
        draw.text((25, ty), c, fill=(30, 30, 30), font=font_bold)

    max_val = max(1, cm.max())
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            x0 = margin_left + j * cell_w
            y0 = margin_top + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            ratio = val / max_val
            # Soft blue gradient
            r = int(240 - ratio * 150)
            g = int(245 - ratio * 120)
            b = int(255 - ratio * 40)
            text_color = (255, 255, 255) if ratio > 0.55 else (20, 20, 20)

            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(200, 200, 200))
            draw.text((x0 + cell_w // 2 - 8, y0 + cell_h // 2 - 8), str(val), fill=text_color, font=font_bold)

    img.save(output_path)


def main():
    splits_path = Path("ml/splits.json")
    if not splits_path.exists():
        raise FileNotFoundError(f"{splits_path} not found. Run training with shared split first.")

    with open(splits_path, "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    test_paths = [Path(p) for p in splits["test"]]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Collect labels for test set
    y_test = []
    for p in test_paths:
        parent_class = p.parent.name
        if parent_class not in class_to_idx:
            raise ValueError(f"Unknown class in test split path: {p}")
        y_test.append(class_to_idx[parent_class])
    y_test = np.array(y_test)

    # Collect junk images
    junk_dir = Path("junk")
    junk_paths = [f for f in junk_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]

    print(f"Loaded shared split: {len(test_paths)} test images across {classes}.")
    print(f"Test class distribution: {np.bincount(y_test)}")
    print(f"Loaded junk images: {len(junk_paths)} images from {junk_dir}/.\n")

    eval_tf = get_eval_transform(224)

    # Pre-load all test images as PIL to keep timing isolated to CPU model inference
    test_tensors = []
    for p in test_paths:
        with Image.open(p) as img:
            test_tensors.append(eval_tf(img.convert("RGB")).unsqueeze(0))

    junk_tensors = []
    for p in junk_paths:
        with Image.open(p) as img:
            junk_tensors.append(eval_tf(img.convert("RGB")).unsqueeze(0))

    results_dir = Path("ml/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_summary": {
            "num_test_images": len(test_paths),
            "num_junk_images": len(junk_paths),
            "classes": classes,
            "test_class_distribution": {classes[i]: int(cnt) for i, cnt in enumerate(np.bincount(y_test))},
        },
        "models": {},
    }

    for cfg in MODEL_CONFIGS:
        mid = cfg["id"]
        w_path = Path(cfg["weights_path"])
        if not w_path.exists():
            print(f"Warning: weights not found for {mid} at {w_path}. Skipping.")
            continue

        print(f"Evaluating {cfg['display_name']} ({w_path})...")
        model, ckpt = load_model(cfg["timm_name"], w_path, len(classes))

        # 1. Parameter count and weights file size
        param_count = sum(p.numel() for p in model.parameters())
        file_size_mb = w_path.stat().st_size / (1024 * 1024)

        # 2. Benchmark CPU inference latency and run test set predictions
        # Warmup passes
        dummy = torch.randn(1, 3, 224, 224)
        for _ in range(5):
            _ = model(dummy)

        test_logits = []
        test_preds = []
        latencies = []

        with torch.no_grad():
            for t in test_tensors:
                t0 = time.perf_counter()
                out = model(t)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms
                logits = out.squeeze(0).numpy()
                test_logits.append(logits)
                test_preds.append(int(logits.argmax()))

        avg_latency_ms = float(np.mean(latencies))
        p95_latency_ms = float(np.percentile(latencies, 95))

        test_logits = np.array(test_logits)  # shape: (N_test, num_classes)
        test_preds = np.array(test_preds)

        # Accuracy & per-class metrics
        acc = float((test_preds == y_test).mean())
        clf_dict = classification_report(y_test, test_preds, target_names=classes, output_dict=True, digits=4)
        cm = confusion_matrix(y_test, test_preds)

        # Save confusion matrix PNG
        cm_png_path = results_dir / f"cm_{mid}.png"
        save_confusion_matrix_image(cm, classes, cm_png_path, f"{cfg['display_name']} Confusion Matrix")

        # 3. Out-Of-Distribution rejection quality with junk/
        junk_logits = []
        with torch.no_grad():
            for jt in junk_tensors:
                out = model(jt)
                junk_logits.append(out.squeeze(0).numpy())
        junk_logits = np.array(junk_logits)

        real_max_logits = test_logits.max(axis=1)
        junk_max_logits = junk_logits.max(axis=1)

        # AUROC separating real flowers (positive class = 1) from junk (negative class = 0)
        y_ood = np.concatenate([np.ones(len(real_max_logits)), np.zeros(len(junk_max_logits))])
        scores_ood = np.concatenate([real_max_logits, junk_max_logits])
        auroc = float(roc_auc_score(y_ood, scores_ood))

        # Threshold keeping 95% of real test images (5th percentile of real max logits)
        tau_95 = float(np.percentile(real_max_logits, 5.0))
        real_retention_at_tau95 = float((real_max_logits >= tau_95).mean())
        # Junk images wrongly accepted: junk with max_logit >= tau_95
        junk_accepted_count = int((junk_max_logits >= tau_95).sum())
        junk_accepted_rate = float(junk_accepted_count / len(junk_max_logits))

        summary["models"][mid] = {
            "name": cfg["display_name"],
            "timm_name": cfg["timm_name"],
            "weights_path": str(w_path),
            "file_size_mb": round(file_size_mb, 2),
            "parameter_count": param_count,
            "avg_cpu_latency_ms": round(avg_latency_ms, 2),
            "p95_cpu_latency_ms": round(p95_latency_ms, 2),
            "accuracy": round(acc, 4),
            "classification_report": clf_dict,
            "confusion_matrix": cm.tolist(),
            "cm_image_path": str(cm_png_path),
            "rejection": {
                "real_max_logit_mean": round(float(real_max_logits.mean()), 3),
                "real_max_logit_std": round(float(real_max_logits.std()), 3),
                "real_max_logit_min": round(float(real_max_logits.min()), 3),
                "junk_max_logit_mean": round(float(junk_max_logits.mean()), 3),
                "junk_max_logit_std": round(float(junk_max_logits.std()), 3),
                "junk_max_logit_max": round(float(junk_max_logits.max()), 3),
                "auroc": round(auroc, 4),
                "tau_95_threshold": round(tau_95, 3),
                "real_retention_at_tau95": round(real_retention_at_tau95, 4),
                "junk_accepted_count": junk_accepted_count,
                "junk_total_count": len(junk_max_logits),
                "junk_accepted_rate": round(junk_accepted_rate, 4),
            },
        }

    # Write comparison.json
    json_path = results_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write comparison.md
    md_path = results_dir / "comparison.md"
    generate_markdown_report(summary, md_path)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    with open(md_path, "r", encoding="utf-8") as f:
        print(f.read())


def generate_markdown_report(summary, md_path):
    dsum = summary.get("dataset_summary", {})
    total_test = dsum.get("num_test_images", 0)
    dist = dsum.get("test_class_distribution", {})
    dist_str = ", ".join(f"{cnt} {cls}" for cls, cnt in dist.items())
    coconut_cnt = dist.get("coconut", 10)
    pct_per_img = (100.0 / total_test) if total_test > 0 else 1.8
    pct_per_coco = (100.0 / coconut_cnt) if coconut_cnt > 0 else 10.0

    lines = []
    lines.append("# Palm Flower Classifier Model Comparison")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Test Split Size & Statistical Context**:")
    lines.append(f"> The held-out test split is small ({total_test} total images: {dist_str}).")
    lines.append(f"> Because the coconut test slice has only {coconut_cnt} images, a single misclassified coconut image changes accuracy by {pct_per_img:.2f}% and class recall by {pct_per_coco:.1f}%.")
    lines.append("> Small differences of a few percent between models are therefore not statistically decisive on their own without cross-validation or a larger test set.")
    lines.append("")

    lines.append("## 1. Overview & Efficiency")
    lines.append("")
    lines.append("| Metric | EfficientNet-B0 | ResNet-18 | ResNet-50 |")
    lines.append("| :--- | :---: | :---: | :---: |")

    eff = summary["models"].get("efficientnet_b0", {})
    r18 = summary["models"].get("resnet18", {})
    r50 = summary["models"].get("resnet50", {})

    lines.append(f"| **Parameters** | {eff.get('parameter_count', 0):,} | {r18.get('parameter_count', 0):,} | {r50.get('parameter_count', 0):,} |")
    lines.append(f"| **Weights Size** | {eff.get('file_size_mb', 0):.2f} MB | {r18.get('file_size_mb', 0):.2f} MB | {r50.get('file_size_mb', 0):.2f} MB |")
    lines.append(f"| **Avg CPU Latency (per img)** | {eff.get('avg_cpu_latency_ms', 0):.1f} ms | {r18.get('avg_cpu_latency_ms', 0):.1f} ms | {r50.get('avg_cpu_latency_ms', 0):.1f} ms |")
    lines.append(f"| **Test Accuracy** | **{eff.get('accuracy', 0)*100:.2f}%** | **{r18.get('accuracy', 0)*100:.2f}%** | **{r50.get('accuracy', 0)*100:.2f}%** |")
    lines.append(f"| **Macro F1 Score** | {eff.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0):.3f} | {r18.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0):.3f} | {r50.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0):.3f} |")
    lines.append("")

    lines.append("## 2. Per-Class Performance")
    lines.append("")
    lines.append("| Model | Class | Precision | Recall | F1-Score | Support |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    for mid, mdata in summary["models"].items():
        cr = mdata.get("classification_report", {})
        mname = mdata["name"]
        for c in summary["dataset_summary"]["classes"]:
            metrics = cr.get(c, {})
            lines.append(
                f"| {mname} | **{c}** | {metrics.get('precision', 0):.3f} | {metrics.get('recall', 0):.3f} | {metrics.get('f1-score', 0):.3f} | {metrics.get('support', 0)} |"
            )
        lines.append(
            f"| *{mname} (Macro Avg)* | *Average* | *{cr.get('macro avg', {}).get('precision', 0):.3f}* | *{cr.get('macro avg', {}).get('recall', 0):.3f}* | *{cr.get('macro avg', {}).get('f1-score', 0):.3f}* | *{cr.get('macro avg', {}).get('support', 0)}* |"
        )
    lines.append("")

    lines.append("## 3. Confusion Matrices")
    lines.append("")
    for mid, mdata in summary["models"].items():
        lines.append(f"### {mdata['name']}")
        lines.append("```")
        classes = summary["dataset_summary"]["classes"]
        lines.append(f"Labels: {classes}")
        cm = np.array(mdata["confusion_matrix"])
        lines.append(str(cm))
        lines.append("```")
        lines.append(f"Saved plot: `ml/results/cm_{mid}.png`\n")

    lines.append("## 4. Rejection Quality (Real vs. Junk Images)")
    lines.append("")
    lines.append("| Metric | EfficientNet-B0 | ResNet-18 | ResNet-50 |")
    lines.append("| :--- | :---: | :---: | :---: |")

    eff_rej = eff.get("rejection", {})
    r18_rej = r18.get("rejection", {})
    r50_rej = r50.get("rejection", {})

    lines.append(f"| **AUROC (Max Logit)** | **{eff_rej.get('auroc', 0):.4f}** | **{r18_rej.get('auroc', 0):.4f}** | **{r50_rej.get('auroc', 0):.4f}** |")
    lines.append(f"| **Real Max Logit (Mean ± Std)** | {eff_rej.get('real_max_logit_mean', 0):.2f} ± {eff_rej.get('real_max_logit_std', 0):.2f} | {r18_rej.get('real_max_logit_mean', 0):.2f} ± {r18_rej.get('real_max_logit_std', 0):.2f} | {r50_rej.get('real_max_logit_mean', 0):.2f} ± {r50_rej.get('real_max_logit_std', 0):.2f} |")
    lines.append(f"| **Junk Max Logit (Mean ± Std)** | {eff_rej.get('junk_max_logit_mean', 0):.2f} ± {eff_rej.get('junk_max_logit_std', 0):.2f} | {r18_rej.get('junk_max_logit_mean', 0):.2f} ± {r18_rej.get('junk_max_logit_std', 0):.2f} | {r50_rej.get('junk_max_logit_mean', 0):.2f} ± {r50_rej.get('junk_max_logit_std', 0):.2f} |")
    lines.append(f"| **Threshold at 95% Real Retention ($\\tau_{{95}}$)** | {eff_rej.get('tau_95_threshold', 0):.3f} | {r18_rej.get('tau_95_threshold', 0):.3f} | {r50_rej.get('tau_95_threshold', 0):.3f} |")
    lines.append(f"| **Junk Wrongly Accepted at $\\tau_{{95}}$** | **{eff_rej.get('junk_accepted_count', 0)} / {eff_rej.get('junk_total_count', 0)}** ({eff_rej.get('junk_accepted_rate', 0)*100:.1f}%) | **{r18_rej.get('junk_accepted_count', 0)} / {r18_rej.get('junk_total_count', 0)}** ({r18_rej.get('junk_accepted_rate', 0)*100:.1f}%) | **{r50_rej.get('junk_accepted_count', 0)} / {r50_rej.get('junk_total_count', 0)}** ({r50_rej.get('junk_accepted_rate', 0)*100:.1f}%) |")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
