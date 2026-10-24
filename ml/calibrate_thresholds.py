"""
Threshold calibration script for palm flower classifier.
Calibrates MAX_LOGIT_MIN and MARGIN_MIN using real test images and junk non-palm images.

Usage:
    python ml/calibrate_thresholds.py --real data/test --junk junk --out backend/thresholds.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate uncertainty rejection thresholds.")
    p.add_argument("--real", default="data/test", help="Path to real test images folder (e.g. data/test/<class>/*.jpg)")
    p.add_argument("--junk", default="junk", help="Path to junk non-palm images folder")
    p.add_argument("--model", default="flower_model.pt", help="Path to trained model checkpoint")
    p.add_argument("--out", default="backend/thresholds.json", help="Path to write thresholds.json")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting junk images")
    p.add_argument("--device", default="cpu", help="Device to run inference on (cpu or cuda)")
    p.add_argument("--min-real-retention", type=float, default=0.80, help="Minimum retention of real test images (default: 0.80)")
    return p.parse_args()


def load_model_and_transform(model_path, device_str):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt.get("classes", ["coconut", "date", "kithul"])
    img_size = ckpt.get("img", 224)

    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return model, eval_tf, classes, device


def infer_image(model, eval_tf, image_path, device, classes):
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            x = eval_tf(rgb_img).unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                top2_probs = torch.topk(probs, 2).values
            
            max_logit = logits.max().item()
            margin = (top2_probs[0] - top2_probs[1]).item()
            pred_class = classes[logits.argmax(1).item()]
            
            return {
                "file": Path(image_path).name,
                "path": str(image_path),
                "max_logit": max_logit,
                "margin": margin,
                "pred_class": pred_class,
                "top_prob": probs.max().item(),
            }
    except Exception as e:
        print(f"Warning: Failed to process image {image_path}: {e}")
        return None


def get_real_image_paths(real_arg):
    real_path = Path(real_arg)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    if real_path.exists():
        files = [p for p in real_path.rglob("*.*") if p.suffix.lower() in valid_exts]
        if files:
            return sorted(files)

    # Fallback to stratified test split of data/ folder (matching training seed=42)
    data_dir = Path("data")
    if data_dir.exists():
        print(f"Notice: '{real_arg}' not found or empty. Using stratified 15% test split from 'data/'...")
        dataset = datasets.ImageFolder("data")
        labels = np.array(dataset.targets)
        idx = np.arange(len(labels))
        tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
        _, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=labels[tmp_idx], random_state=42)
        
        fallback_files = [Path(dataset.samples[i][0]) for i in te_idx]
        return sorted(fallback_files)

    raise FileNotFoundError(f"Could not locate real test images at '{real_arg}' or 'data/'.")


def get_junk_image_paths(junk_arg):
    junk_path = Path(junk_arg)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not junk_path.exists():
        raise FileNotFoundError(f"Junk non-palm images directory not found at: {junk_arg}")
    
    files = [p for p in junk_path.rglob("*.*") if p.suffix.lower() in valid_exts]
    if not files:
        raise ValueError(f"No valid images found in junk directory: {junk_arg}")
    
    return sorted(files)


def main():
    args = parse_args()
    print("=" * 65)
    print("      PALM FLOWER CLASSIFIER - THRESHOLD CALIBRATION")
    print("=" * 65)

    model, eval_tf, classes, device = load_model_and_transform(args.model, args.device)
    print(f"Model loaded from {args.model} on {device}")
    print(f"Classes: {classes}")

    # 1. Load real test images
    real_paths = get_real_image_paths(args.real)
    print(f"Evaluating {len(real_paths)} real palm flower images...")
    real_results = []
    for p in real_paths:
        res = infer_image(model, eval_tf, p, device, classes)
        if res is not None:
            real_results.append(res)

    real_logits = np.array([r["max_logit"] for r in real_results])
    real_margins = np.array([r["margin"] for r in real_results])

    # 2. Load junk non-palm images
    junk_paths = get_junk_image_paths(args.junk)
    print(f"Evaluating {len(junk_paths)} junk non-palm images from '{args.junk}'...")
    junk_results = []
    for p in junk_paths:
        res = infer_image(model, eval_tf, p, device, classes)
        if res is not None:
            junk_results.append(res)

    # 3. Print min, median, max for each group
    print("\n" + "-" * 65)
    print("1. SUMMARY STATISTICS FOR REAL AND JUNK GROUPS")
    print("-" * 65)
    all_junk_logits = np.array([j["max_logit"] for j in junk_results])
    all_junk_margins = np.array([j["margin"] for j in junk_results])

    print("REAL PALM IMAGES (N = {}):".format(len(real_results)))
    print("  Max Logit:  min = {:+0.4f},  median = {:+0.4f},  max = {:+0.4f}".format(
        real_logits.min(), np.median(real_logits), real_logits.max()))
    print("  Margin:     min = {:0.4f},   median = {:0.4f},   max = {:0.4f}".format(
        real_margins.min(), np.median(real_margins), real_margins.max()))

    print("\nALL JUNK NON-PALM IMAGES (N = {}):".format(len(junk_results)))
    print("  Max Logit:  min = {:+0.4f},  median = {:+0.4f},  max = {:+0.4f}".format(
        all_junk_logits.min(), np.median(all_junk_logits), all_junk_logits.max()))
    print("  Margin:     min = {:0.4f},   median = {:0.4f},   max = {:0.4f}".format(
        all_junk_margins.min(), np.median(all_junk_margins), all_junk_margins.max()))

    # 4. Split junk images 50/50 into tuning half and check half (fixed random seed)
    rng = random.Random(args.seed)
    shuffled_junk = list(junk_results)
    rng.shuffle(shuffled_junk)
    
    half = len(shuffled_junk) // 2
    tuning_junk = shuffled_junk[:half]
    check_junk = shuffled_junk[half:]

    tuning_logits = np.array([j["max_logit"] for j in tuning_junk])
    tuning_margins = np.array([j["margin"] for j in tuning_junk])

    print("\n" + "-" * 65)
    print(f"2. JUNK SPLIT: Tuning Half ({len(tuning_junk)}) & Check Half ({len(check_junk)})")
    print("-" * 65)
    print("TUNING HALF JUNK (N = {}):".format(len(tuning_junk)))
    print("  Max Logit:  min = {:+0.4f},  median = {:+0.4f},  max = {:+0.4f}".format(
        tuning_logits.min(), np.median(tuning_logits), tuning_logits.max()))
    print("  Margin:     min = {:0.4f},   median = {:0.4f},   max = {:0.4f}".format(
        tuning_margins.min(), np.median(tuning_margins), tuning_margins.max()))

    # 5. Choose MAX_LOGIT_MIN and MARGIN_MIN
    # If a clean gap exists between real and junk, use the gap midpoint.
    # When overlap exists, optimize (max_logit, margin) to maximize junk rejection while
    # preserving at least args.min_real_retention of real images.
    real_logit_min = float(real_logits.min())
    tuning_logit_max = float(tuning_logits.max())
    real_margin_min = float(real_margins.min())
    tuning_margin_max = float(tuning_margins.max())
    
    if real_logit_min > tuning_logit_max and real_margin_min > tuning_margin_max:
        chosen_max_logit = round((real_logit_min + tuning_logit_max) / 2.0, 4)
        chosen_margin = round((real_margin_min + tuning_margin_max) / 2.0, 4)
        method_desc = f"Clean gap separation between real min and tuning max"
    else:
        # Overlap optimization: grid search over candidate thresholds
        best_score = -1
        best_pair = (0.60, 0.30)

        cand_logits = np.linspace(np.percentile(real_logits, 2), np.percentile(real_logits, 40), 40)
        cand_margins = np.linspace(np.percentile(real_margins, 2), np.percentile(real_margins, 40), 40)

        for l_cand in cand_logits:
            for m_cand in cand_margins:
                real_kept = np.mean((real_logits >= l_cand) & (real_margins >= m_cand))
                if real_kept >= args.min_real_retention:
                    junk_rej = np.mean((tuning_logits < l_cand) | (tuning_margins < m_cand))
                    score = real_kept + 1.5 * junk_rej
                    if score > best_score:
                        best_score = score
                        best_pair = (round(float(l_cand), 4), round(float(m_cand), 4))

        chosen_max_logit, chosen_margin = best_pair
        method_desc = f"Overlap optimization (target real retention >= {args.min_real_retention*100:.0f}%, maximizing junk rejection)"

    print("\n" + "-" * 65)
    print("3. CHOSEN THRESHOLDS")
    print("-" * 65)
    print(f"  MAX_LOGIT_MIN = {chosen_max_logit:.4f}  ({method_desc})")
    print(f"  MARGIN_MIN    = {chosen_margin:.4f}  ({method_desc})")

    # 6. Evaluation function: An image is rejected if max_logit < MAX_LOGIT_MIN or margin < MARGIN_MIN
    def evaluate_rejection(sample_list):
        rejected = [s for s in sample_list if (s["max_logit"] < chosen_max_logit or s["margin"] < chosen_margin)]
        accepted = [s for s in sample_list if (s["max_logit"] >= chosen_max_logit and s["margin"] >= chosen_margin)]
        return rejected, accepted

    real_rejected, real_accepted = evaluate_rejection(real_results)
    tuning_rejected, tuning_accepted = evaluate_rejection(tuning_junk)
    check_rejected, check_accepted = evaluate_rejection(check_junk)

    print("\n" + "-" * 65)
    print("4. CALIBRATION PERFORMANCE REPORT")
    print("-" * 65)
    print("Real Palm Test Set (Total: {}):".format(len(real_results)))
    print("  - Correctly Accepted: {:3d} / {} ({:5.1f}%)".format(
        len(real_accepted), len(real_results), (len(real_accepted) / len(real_results)) * 100))
    print("  - Wrongly Rejected:   {:3d} / {} ({:5.1f}%)".format(
        len(real_rejected), len(real_results), (len(real_rejected) / len(real_results)) * 100))

    print("\nTuning-Half Junk (Total: {}):".format(len(tuning_junk)))
    print("  - Correctly Rejected: {:3d} / {} ({:5.1f}%)".format(
        len(tuning_rejected), len(tuning_junk), (len(tuning_rejected) / len(tuning_junk)) * 100))
    print("  - Wrongly Accepted:   {:3d} / {} ({:5.1f}%)".format(
        len(tuning_accepted), len(tuning_junk), (len(tuning_accepted) / len(tuning_junk)) * 100))

    print("\nCheck-Half Junk (Total: {}):".format(len(check_junk)))
    print("  - Correctly Rejected: {:3d} / {} ({:5.1f}%)".format(
        len(check_rejected), len(check_junk), (len(check_rejected) / len(check_junk)) * 100))
    print("  - Wrongly Accepted:   {:3d} / {} ({:5.1f}%)".format(
        len(check_accepted), len(check_junk), (len(check_accepted) / len(check_junk)) * 100))

    # 7. Write to backend/thresholds.json
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_dict = {
        "MAX_LOGIT_MIN": chosen_max_logit,
        "MARGIN_MIN": chosen_margin
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"\nWrote calibrated thresholds to: {out_path.resolve()}")

    # 8. List filenames of any junk images that were wrongly accepted
    all_wrongly_accepted = tuning_accepted + check_accepted
    print("\n" + "-" * 65)
    print("5. JUNK IMAGES WRONGLY ACCEPTED (Total: {})".format(len(all_wrongly_accepted)))
    print("-" * 65)
    if all_wrongly_accepted:
        print("{:<30} {:<10} {:<12} {:<10} {:<12}".format("Filename", "Split", "Max Logit", "Margin", "Predicted"))
        print("-" * 78)
        for s in tuning_accepted:
            print("{:<30} {:<10} {:<12.4f} {:<10.4f} {:<12}".format(
                s["file"][:28], "tuning", s["max_logit"], s["margin"], s["pred_class"]))
        for s in check_accepted:
            print("{:<30} {:<10} {:<12.4f} {:<10.4f} {:<12}".format(
                s["file"][:28], "check", s["max_logit"], s["margin"], s["pred_class"]))
    else:
        print("None! All junk non-palm images were successfully rejected as uncertain.")

    print("=" * 65)


if __name__ == "__main__":
    main()
