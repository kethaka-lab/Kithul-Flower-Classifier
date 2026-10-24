"""
Flower classifier API.

Run from the project root (with the venv active):
    python -m uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000/docs and use POST /predict to upload an image.
"""
import io
import json
import os
from pathlib import Path

import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Model file lives in the project root by default; override with MODEL_PATH.
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "flower_model.pt"))
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


@app.on_event("startup")
def load_model():
    global model, classes, transform
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Train the model first or set MODEL_PATH.")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]
    img = ckpt.get("img", 224)
    # pretrained=False: weights come from our checkpoint, so nothing is downloaded.
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    # Must match the evaluation transform used in training.
    transform = transforms.Compose([
        transforms.Resize(int(img * 1.14)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

@app.get("/health")
def health():
    return {"status": "ok", "classes": classes, "device": device}


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


# Mount the frontend directory so visiting http://127.0.0.1:8000 loads the demo UI
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

