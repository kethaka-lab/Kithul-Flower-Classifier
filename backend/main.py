import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Model file lives in the project root by default; override with MODEL_PATH.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "flower_model.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="Palm Flower Classifier")

# Allow the frontend (any origin for now) to call the API. Tighten this before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if any(request.url.path.endswith(ext) for ext in [".html", ".js", ".css"]) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
classes = []
transform = None

# Multi-model storage for /predict_all
MULTI_MODELS = {}

MODEL_ARCH_MAP = {
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
}


@app.on_event("startup")
def load_all_models():
    global model, classes, transform, MULTI_MODELS
    # 1. Existing legacy model loader for /predict
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Train the model first or set MODEL_PATH.")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]
    img = ckpt.get("img", 224)
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(int(img * 1.14)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 2. Multi-model loader for /predict_all
    weights_dir = PROJECT_ROOT / "ml" / "weights"
    rejection_dir = PROJECT_ROOT / "backend" / "rejection"

    for mid in ["efficientnet_b0", "resnet18", "resnet50"]:
        w_path = weights_dir / f"{mid}.pt"
        r_path = rejection_dir / f"{mid}.json"
        if not w_path.exists():
            print(f"Warning: Weights not found for {mid} at {w_path}")
            continue

        m_ckpt = torch.load(w_path, map_location=device)
        m_name = m_ckpt.get("model_name", mid)
        m_classes = m_ckpt.get("classes", classes)
        timm_arch = MODEL_ARCH_MAP.get(m_name, m_name)

        m_instance = timm.create_model(timm_arch, pretrained=False, num_classes=len(m_classes))
        m_instance.load_state_dict(m_ckpt["state_dict"])
        m_instance.to(device).eval()

        rej_data = {}
        if r_path.exists():
            with open(r_path, "r", encoding="utf-8") as f:
                rej_data = json.load(f)

        centroids = np.array(rej_data.get("centroids", []), dtype=np.float32)
        tau_feat = float(rej_data.get("tau_feat", 0.65))

        MULTI_MODELS[mid] = {
            "model": m_instance,
            "classes": m_classes,
            "centroids": centroids,
            "tau_feat": tau_feat,
            "tau_feat_95": float(rej_data.get("tau_feat_95", tau_feat)),
            "tau_feat_90": float(rej_data.get("tau_feat_90", tau_feat)),
        }
    print(f"Loaded {len(MULTI_MODELS)} comparison models: {list(MULTI_MODELS.keys())}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": classes,
        "device": device,
        "multi_models": list(MULTI_MODELS.keys()),
    }


DEFAULT_THRESHOLDS = {"MAX_LOGIT_MIN": 0.60, "MARGIN_MIN": 0.20}


def get_thresholds():
    """Load MAX_LOGIT_MIN and MARGIN_MIN from config file (backend/thresholds.json) with safe defaults."""
    candidates = [
        Path(__file__).resolve().parent / "thresholds.json",
        MODEL_PATH.parent / "thresholds.json",
        MODEL_PATH.parent / "backend" / "thresholds.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        "MAX_LOGIT_MIN": float(data.get("MAX_LOGIT_MIN", DEFAULT_THRESHOLDS["MAX_LOGIT_MIN"])),
                        "MARGIN_MIN": float(data.get("MARGIN_MIN", DEFAULT_THRESHOLDS["MARGIN_MIN"])),
                    }
            except Exception:
                pass
    return DEFAULT_THRESHOLDS


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file is not a readable image.")

    x = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs_tensor = torch.softmax(logits, dim=1)[0]
        top2 = torch.topk(probs_tensor, 2).values
        probs = probs_tensor.cpu().tolist()

    max_logit = logits.max().item()
    margin = (top2[0] - top2[1]).item()

    ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
    top_label = ranked[0][0]

    thresholds = get_thresholds()
    max_logit_min = thresholds["MAX_LOGIT_MIN"]
    margin_min = thresholds["MARGIN_MIN"]

    if max_logit < max_logit_min or margin < margin_min:
        status = "uncertain"
        label = None
    else:
        status = "ok"
        label = top_label

    return {
        "status": status,
        "label": label,
        "prediction": label if label is not None else "uncertain",
        "confident": (status == "ok"),
        "probabilities": {cls: round(p, 4) for cls, p in ranked},
        "max_logit": round(max_logit, 4),
        "margin": round(margin, 4),
    }


@app.post("/predict_all")
async def predict_all(file: UploadFile = File(...)):
    """Run inference and cosine-centroid rejection across efficientnet_b0, resnet18, and resnet50."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file is not a readable image.")

    x = transform(image).unsqueeze(0).to(device)
    results = []

    for mid in ["efficientnet_b0", "resnet18", "resnet50"]:
        if mid not in MULTI_MODELS:
            continue
        m_info = MULTI_MODELS[mid]
        m = m_info["model"]
        m_classes = m_info["classes"]
        centroids = m_info["centroids"]
        tau_feat = m_info["tau_feat"]

        t0 = time.perf_counter()
        with torch.no_grad():
            feat_map = m.forward_features(x)
            feat_tensor = m.forward_head(feat_map, pre_logits=True)
            logits_tensor = m.forward_head(feat_map, pre_logits=False)
            probs_tensor = torch.softmax(logits_tensor, dim=1)[0]
            probs_list = probs_tensor.cpu().tolist()
        t1 = time.perf_counter()
        inference_ms = round((t1 - t0) * 1000.0, 2)

        feat_np = feat_tensor.squeeze(0).cpu().numpy()
        # Cosine similarity to class centroids
        if len(centroids) > 0:
            feat_norm = feat_np / (np.linalg.norm(feat_np) + 1e-10)
            cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
            cos_sims = np.dot(cent_norm, feat_norm)
            max_cosine = float(cos_sims.max())
        else:
            max_cosine = 1.0

        ranked = sorted(zip(m_classes, probs_list), key=lambda p: p[1], reverse=True)
        top_label = ranked[0][0]

        is_ok = max_cosine >= tau_feat
        status = "ok" if is_ok else "uncertain"
        label = top_label if is_ok else None

        results.append({
            "model": mid,
            "status": status,
            "label": label,
            "probabilities": {cls: round(p, 4) for cls, p in ranked},
            "max_cosine": round(max_cosine, 4),
            "tau_feat": round(tau_feat, 4),
            "inference_ms": inference_ms,
        })

    return {"predictions": results}


@app.get("/results")
def get_results():
    """Merge comparison.json and rejection_scores.json."""
    results_dir = PROJECT_ROOT / "ml" / "results"
    comp_file = results_dir / "comparison.json"
    rej_file = results_dir / "rejection_scores.json"

    comp_data = {}
    rej_data = {}

    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
    if rej_file.exists():
        with open(rej_file, "r", encoding="utf-8") as f:
            rej_data = json.load(f)

    return {
        "comparison": comp_data,
        "rejection_scores": rej_data,
    }


@app.get("/results/image/{name}")
def get_result_image(name: str):
    """Serve confusion matrix PNGs from ml/results/ (safe regex, no path traversal)."""
    if not re.match(r"^cm_[a-zA-Z0-9_\-]+\.png$", name):
        raise HTTPException(status_code=400, detail="Invalid image name format.")

    img_path = (PROJECT_ROOT / "ml" / "results" / name).resolve()
    expected_dir = (PROJECT_ROOT / "ml" / "results").resolve()

    if expected_dir not in img_path.parents or not img_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(str(img_path), media_type="image/png")


# Mount the frontend directory so visiting http://127.0.0.1:8000 loads the demo UI
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


