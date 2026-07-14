"""Load best_model_full.pt and score a single face image."""

from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_architecture import SmallResNet  # noqa: E402

MODEL_PATH = ROOT / "best_model_full.pt"
CASCADE_PATH = Path(__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"

# Empirical FFHQ label range from the project dataset (REPORT.md §6).
SCORE_LO = 11.2
SCORE_HI = 33.3
MODEL_MAE = 1.32

# Expand Haar face boxes so the crop looks closer to FFHQ framing
# (tight portrait with hair/shoulders, face filling most of the frame).
FACE_MARGIN = 0.55


@dataclass(frozen=True)
class ScoreResult:
    score: float
    band: str
    band_summary: str
    band_detail: str
    percentile: float
    lo: float
    hi: float
    mae: float
    epoch: int | str
    crop_method: str
    crop_note: str
    crop_preview_b64: str


def _band_for_score(score: float) -> tuple[str, str, str]:
    if score < 16:
        return (
            "Very low quality",
            "Likely unsuitable for reliable biometric matching.",
            "Scores in this range often indicate serious issues such as heavy blur, "
            "extreme pose, poor lighting, or significant occlusion — or that the face "
            "crop still doesn't look like an FFHQ-style portrait.",
        )
    if score < 20:
        return (
            "Below average",
            "Noticeable quality limitations for face recognition use.",
            "The image may have moderate blur, off-angle pose, uneven illumination, "
            "or partial occlusion. Usable in some settings, but not ideal for "
            "high-assurance biometric systems.",
        )
    if score < 26:
        return (
            "Average / typical",
            "This is where most faces land — not a bad score.",
            "On the FFHQ training set, the majority of predictions fall in the low-to-mid "
            "20s. An 'average' band here means typical biometric usability, not a failed "
            "photo. Scores above ~28 are rare even for sharp, frontal faces.",
        )
    if score < 29:
        return (
            "Above average",
            "Good biometric usability — clear and well-formed.",
            "Higher than most of the training distribution. Expect good sharpness, "
            "reasonable pose, and balanced lighting.",
        )
    return (
        "High quality",
        "Among the best-scoring faces in the dataset.",
        "Only about 1% of training labels exceeded 30. Images here are likely sharp, "
        "frontal, evenly lit, and largely unobstructed.",
    )


def _percentile(score: float, lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 0:
        return 50.0
    return max(0.0, min(100.0, (score - lo) / span * 100.0))


def _square_crop_around(img: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    img_w, img_h = img.size
    side = min(float(side), float(img_w), float(img_h))
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = max(0, min(left, img_w - int(side)))
    top = max(0, min(top, img_h - int(side)))
    return img.crop((left, top, left + int(side), top + int(side)))


def _center_square_crop(img: Image.Image, upper_bias: float = 0.0) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    # upper_bias > 0 shifts the crop toward the top (typical portrait faces)
    top = int((h - side) * max(0.0, min(1.0, upper_bias)))
    top = max(0, min(top, h - side))
    return img.crop((left, top, left + side, top + side))


def _haar_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return face boxes via OpenCV Haar, or [] if OpenCV is broken/unavailable.

    Streamlit Cloud sometimes ships a stub ``cv2`` without CascadeClassifier.
    We never raise from here — callers fall back to other crop strategies.
    """
    try:
        import cv2  # local import so a broken cv2 can't break module import
    except Exception:
        return []

    classifier_cls = getattr(cv2, "CascadeClassifier", None)
    if classifier_cls is None:
        return []

    cascade_file = CASCADE_PATH
    if not cascade_file.exists():
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if data_dir:
            cascade_file = Path(data_dir) / "haarcascade_frontalface_default.xml"
    if not cascade_file.exists():
        return []

    try:
        cascade = classifier_cls(str(cascade_file))
        if cascade.empty() if hasattr(cascade, "empty") else False:
            return []
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
    except Exception:
        return []

    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _skin_face_box(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Approximate a face box from skin-colored pixels (no OpenCV required)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    # Classic RGB skin heuristic
    skin = (
        (r > 95) & (g > 40) & (b > 20)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
        & (np.abs(r - g) > 15) & (r > g) & (r > b)
    )
    ys, xs = np.where(skin)
    if len(xs) < max(200, rgb.shape[0] * rgb.shape[1] * 0.005):
        return None

    # Use central mass of skin pixels, ignore sparse outliers via percentiles
    x0, x1 = np.percentile(xs, [8, 92]).astype(int)
    y0, y1 = np.percentile(ys, [5, 90]).astype(int)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    # Reject if the "face" is basically the whole image or tiny
    area = w * h
    img_area = rgb.shape[0] * rgb.shape[1]
    if area < 0.02 * img_area or area > 0.85 * img_area:
        return None
    return int(x0), int(y0), int(w), int(h)


def _face_square_crop(img: Image.Image) -> tuple[Image.Image, str, str]:
    """Crop a face-centered square similar to FFHQ training images."""
    rgb = np.array(img.convert("RGB"))
    # Luma for Haar without requiring cv2.cvtColor
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)

    faces = _haar_faces(gray)
    if faces:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * (1.0 + 2.0 * FACE_MARGIN)
        crop = _square_crop_around(img, cx, cy, side)
        n = len(faces)
        note = (
            f"Detected {n} face{'s' if n != 1 else ''}; scored the largest face after an "
            "FFHQ-style square crop. The model only saw tightly cropped faces during training."
        )
        return crop, "face_detected", note

    skin = _skin_face_box(rgb)
    if skin is not None:
        x, y, w, h = skin
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * (1.0 + 2.0 * FACE_MARGIN)
        crop = _square_crop_around(img, cx, cy, side)
        return (
            crop,
            "skin_estimate",
            "OpenCV face detector unavailable — used a skin-tone estimate to crop. "
            "Prefer a close-up face photo for best results.",
        )

    crop = _center_square_crop(img, upper_bias=0.22)
    return (
        crop,
        "center_square",
        "No face detected — used an upper-biased center crop. Scores are unreliable if "
        "the face is off-center or small in the frame. Prefer a close-up face photo.",
    )


def _pil_to_b64_jpeg(img: Image.Image, size: int = 256) -> str:
    preview = img.convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class QualityScorer:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.lo = float(ckpt.get("target_lo", SCORE_LO))
        self.hi = float(ckpt.get("target_hi", SCORE_HI))
        self.head_act = ckpt.get("head_act", "sigmoid")
        self.epoch = ckpt.get("epoch", "?")

        n_aux = len(ckpt.get("aux_cols") or [])
        self.model = SmallResNet(
            head_act=self.head_act,
            n_aux=n_aux,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def score_image_bytes(self, data: bytes) -> ScoreResult:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        crop, crop_method, crop_note = _face_square_crop(img)
        tensor = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            if isinstance(out, tuple):
                out = out[0]
            raw = float(out.squeeze().cpu())

        if self.head_act == "linear":
            raw = max(0.0, min(1.0, raw))

        native = raw * (self.hi - self.lo) + self.lo
        band, summary, detail = _band_for_score(native)

        return ScoreResult(
            score=round(native, 2),
            band=band,
            band_summary=summary,
            band_detail=detail,
            percentile=round(_percentile(native, self.lo, self.hi), 1),
            lo=round(self.lo, 2),
            hi=round(self.hi, 2),
            mae=MODEL_MAE,
            epoch=self.epoch,
            crop_method=crop_method,
            crop_note=crop_note,
            crop_preview_b64=_pil_to_b64_jpeg(crop),
        )
