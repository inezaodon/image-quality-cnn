"""Load best_model_full.pt and score a single face image."""

from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_architecture import SmallResNet  # noqa: E402

MODEL_PATH = ROOT / "best_model_full.pt"

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


def _center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _face_square_crop(img: Image.Image) -> tuple[Image.Image, str, str]:
    """Crop a face-centered square similar to FFHQ training images.

    The model was trained only on tightly aligned FFHQ face crops. Full-body
    or scenic photos with a small face score very low unless we crop first.
    """
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    if len(faces) == 0:
        crop = _center_square_crop(img)
        return (
            crop,
            "center_square",
            "No face detected — used a center square crop. Scores are unreliable if "
            "the face is off-center or small in the frame. Prefer a close-up face photo.",
        )

    # Largest face by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1.0 + 2.0 * FACE_MARGIN)
    img_w, img_h = img.size
    side = min(side, img_w, img_h)

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = max(0, min(left, img_w - int(side)))
    top = max(0, min(top, img_h - int(side)))
    right = left + int(side)
    bottom = top + int(side)

    crop = img.crop((left, top, right, bottom))
    n = len(faces)
    note = (
        f"Detected {n} face{'s' if n != 1 else ''}; scored the largest face after an "
        "FFHQ-style square crop. The model only saw tightly cropped faces during training."
    )
    return crop, "face_detected", note


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

        # Match evaluate.py: ToTensor + Normalize. Resize after face crop so
        # arbitrary phone/web photos become 256×256 FFHQ-like inputs.
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def score_image_bytes(self, data: bytes) -> ScoreResult:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # EXIF orientation (phone photos often look upright in viewers but
        # arrive rotated to the model without this).
        try:
            from PIL import ImageOps
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
