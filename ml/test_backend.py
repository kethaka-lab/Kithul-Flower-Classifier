"""
Test script for verifying backend endpoints on a running server.
Sends:
- 2 real test images per class from ml/splits.json (6 images total)
- Every image in junk/
- Every image in fresh_junk/ (if directory exists)

Prints:
- Detailed table of prediction status, label, max cosine, and inference latency per image per model
- Overall summary of real flowers accepted and junk images rejected per model
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests is not installed in this environment. Please run: pip install requests")
    sys.exit(1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Test backend API with real test and junk images.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of running server")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Check server health
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        print(f"Connected to backend: {base_url}")
        print(f"Active models on server: {health.get('multi_models', [])}\n")
    except Exception as e:
        print(f"Error: Could not reach backend server at {base_url}: {e}")
        print("Please start the backend server first:")
        print("    python -m uvicorn backend.main:app --port 8000")
        sys.exit(1)

    splits_path = Path("ml/splits.json")
    if not splits_path.exists():
        print(f"Error: {splits_path} not found.")
        sys.exit(1)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # 1. Select 2 real test images per class
    classes = splits.get("classes", ["coconut", "date", "kithul"])
    test_paths = [Path(p) for p in splits.get("test", [])]

    selected_real = []
    for c in classes:
        c_paths = [p for p in test_paths if p.parent.name == c]
        selected_real.extend(c_paths[:2])

    # 2. Collect junk images
    junk_dir = Path("junk")
    junk_images = sorted([f for f in junk_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS])

    # 3. Collect fresh_junk images (if folder exists)
    fresh_junk_dir = Path("fresh_junk")
    fresh_junk_images = []
    if fresh_junk_dir.exists():
        fresh_junk_images = sorted([f for f in fresh_junk_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS])

    test_queue = []
    for p in selected_real:
        test_queue.append(("REAL", p.parent.name, p))
    for p in junk_images:
        test_queue.append(("JUNK", "junk", p))
    for p in fresh_junk_images:
        test_queue.append(("FRESH_JUNK", "fresh_junk", p))

    print(f"Test Queue: {len(selected_real)} real flower images, {len(junk_images)} junk images, {len(fresh_junk_images)} fresh_junk images.")
    print("=" * 115)
    header = f"{'Type':<11} | {'Image':<32} | {'Model':<15} | {'Status':<9} | {'Label':<8} | {'Max Cos':<8} | {'Tau':<7} | {'Time (ms)'}"
    print(header)
    print("-" * 115)

    stats = defaultdict(lambda: {
        "real_total": 0, "real_accepted": 0,
        "junk_total": 0, "junk_rejected": 0,
        "fresh_junk_total": 0, "fresh_junk_rejected": 0,
    })

    for img_type, true_category, p in test_queue:
        with open(p, "rb") as f:
            resp = requests.post(
                f"{base_url}/predict_all",
                files={"file": (p.name, f, "image/jpeg")},
                timeout=10,
            )
        if resp.status_code != 200:
            print(f"Error {resp.status_code} for {p}: {resp.text}")
            continue

        preds = resp.json().get("predictions", [])
        display_name = p.name if len(p.name) <= 30 else (p.name[:14] + "..." + p.name[-13:])

        for idx, pred in enumerate(preds):
            m = pred["model"]
            st = pred["status"]
            lbl = str(pred["label"])
            cos = pred["max_cosine"]
            tau = pred["tau_feat"]
            t_ms = pred["inference_ms"]

            t_col = img_type if idx == 0 else ""
            name_col = display_name if idx == 0 else ""

            print(f"{t_col:<11} | {name_col:<32} | {m:<15} | {st:<9} | {lbl:<8} | {cos:<8.4f} | {tau:<7.4f} | {t_ms:<6.1f}")

            # Update stats
            if img_type == "REAL":
                stats[m]["real_total"] += 1
                if st == "ok":
                    stats[m]["real_accepted"] += 1
            elif img_type == "JUNK":
                stats[m]["junk_total"] += 1
                if st == "uncertain":
                    stats[m]["junk_rejected"] += 1
            elif img_type == "FRESH_JUNK":
                stats[m]["fresh_junk_total"] += 1
                if st == "uncertain":
                    stats[m]["fresh_junk_rejected"] += 1

        print("-" * 115)

    print("\n" + "=" * 80)
    print("BACKEND EVALUATION SUMMARY")
    print("=" * 80)
    summary_header = f"{'Model':<16} | {'Real Flowers Accepted':<25} | {'Junk Rejected':<20}"
    if fresh_junk_images:
        summary_header += f" | {'Fresh Junk Rejected':<20}"
    print(summary_header)
    print("-" * len(summary_header))

    for m in ["efficientnet_b0", "resnet18", "resnet50"]:
        if m not in stats:
            continue
        s = stats[m]
        r_acc = f"{s['real_accepted']}/{s['real_total']} ({s['real_accepted']/s['real_total']*100:.1f}%)" if s['real_total'] else "N/A"
        j_rej = f"{s['junk_rejected']}/{s['junk_total']} ({s['junk_rejected']/s['junk_total']*100:.1f}%)" if s['junk_total'] else "N/A"
        row = f"{m:<16} | {r_acc:<25} | {j_rej:<20}"
        if fresh_junk_images:
            fj_rej = f"{s['fresh_junk_rejected']}/{s['fresh_junk_total']} ({s['fresh_junk_rejected']/s['fresh_junk_total']*100:.1f}%)" if s['fresh_junk_total'] else "N/A"
            row += f" | {fj_rej:<20}"
        print(row)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
