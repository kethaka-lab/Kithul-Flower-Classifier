"""
Flower classifier API.

Run from the project root (with the venv active):
    python -m uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000/docs and use POST /predict to upload an image.
"""
import io
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
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="Palm Flower Classifier")

# Allow the frontend (any origin for now) to call the API. Tighten this before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Kithul Flower Classifier API!",
        "docs": "/docs",
        "health": "/health"
    }
    
@app.get("/health")
def health():
    return {"status": "ok", "classes": classes, "device": device}


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
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()

    ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
    top_label, top_conf = ranked[0]
    return {
        "prediction": top_label,
        "confidence": round(top_conf, 4),
        "confident": top_conf >= CONFIDENCE_THRESHOLD,
        "probabilities": {label: round(p, 4) for label, p in ranked},
    }
