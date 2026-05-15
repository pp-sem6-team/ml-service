import os
import io
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import timm
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms


# Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Config (из переменных окружения)

MODEL_PATH = os.getenv("MODEL_PATH", "models/efficientnet_merged.pth")
API_KEY = os.getenv("API_KEY", "changeme-secret-key")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["combination", "dry", "normal", "oily"]
IMG_SIZE = 224
LOW_CONF_THRESHOLD = 0.60


# Preprocessing

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def detect_and_crop_face(img: Image.Image, pad: float = 0.2) -> Image.Image | None:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_px = int(pad * w)
    x1 = max(0, x - pad_px)
    y1 = max(0, y - pad_px)
    x2 = min(arr.shape[1], x + w + pad_px)
    y2 = min(arr.shape[0], y + h + pad_px)
    return Image.fromarray(arr[y1:y2, x1:x2])


_clahe = CLAHETransform()

_normalize = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)

tta_transforms = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), _clahe, transforms.ToTensor(), _normalize]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), _clahe, transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _normalize]),
    transforms.Compose([transforms.Resize((int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1))), _clahe, transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), _normalize]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), _clahe, transforms.RandomRotation(degrees=(10, 10)), transforms.ToTensor(), _normalize]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), _clahe, transforms.RandomRotation(degrees=(-10, -10)), transforms.ToTensor(), _normalize]),
]


# Model

model: torch.nn.Module | None = None


def load_model() -> torch.nn.Module:
    logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}")
    m = timm.create_model("efficientnet_b2", pretrained=False, num_classes=4)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    logger.info("Model loaded successfully")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None



# App

app = FastAPI(
    title="Skin Type Classifier",
    version=MODEL_VERSION,
    lifespan=lifespan,
)



# Schemas

class PredictResponse(BaseModel):
    model_version: str
    skin_type: str
    confidence: float
    probabilities: dict[str, float]
    warnings: list[str]



# Auth helper

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")



# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
async def predict_skin_type(
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    # Auth
    verify_api_key(x_api_key)

    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpeg/png/webp)")

    # Read image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file")

    # Validate size
    if image.width < 64 or image.height < 64:
        raise HTTPException(status_code=422, detail="Image is too small (minimum 64x64)")

    warnings: list[str] = []

    # Face detection
    face = detect_and_crop_face(image)
    if face is None:
        raise HTTPException(status_code=422, detail="No face detected in the image")

    # Inference with TTA
    try:
        with torch.no_grad():
            probs_list = []
            for tf in tta_transforms:
                tensor = tf(face).unsqueeze(0).to(DEVICE)
                out = torch.softmax(model(tensor), dim=1).cpu().numpy()
                probs_list.append(out)

        avg_probs = np.mean(probs_list, axis=0)[0]
        pred_idx  = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])
        skin_type  = CLASSES[pred_idx]

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    # Warnings
    if confidence < LOW_CONF_THRESHOLD:
        warnings.append(
            f"Low confidence ({confidence:.2f}). "
            "Try a clearer, well-lit photo of your face."
        )

    return PredictResponse(
        model_version=MODEL_VERSION,
        skin_type=skin_type,
        confidence=round(confidence, 4),
        probabilities={cls: round(float(p), 4) for cls, p in zip(CLASSES, avg_probs)},
        warnings=warnings,
    )
