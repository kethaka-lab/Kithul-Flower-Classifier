"""
Common utilities for feature extraction, model loading, and rejection scoring.
"""
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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


def get_eval_transform(img_size=224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_model(timm_name, weights_path, num_classes=3):
    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


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


def compute_class_centroids(features, labels, num_classes):
    centroids = []
    for c_idx in range(num_classes):
        mask = (labels == c_idx)
        if mask.sum() > 0:
            centroids.append(features[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(features.shape[1], dtype=np.float32))
    return np.array(centroids)


def compute_cosine_similarity(features, centroids):
    # features: (N, D), centroids: (K, D)
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    cos_sim = np.dot(feat_norm, cent_norm.T)  # (N, K)
    return cos_sim.max(axis=1)  # max cosine to nearest class centroid
