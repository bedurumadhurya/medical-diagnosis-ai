from __future__ import annotations

import io

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


CHEST_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

MRI_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]


def load_image_bytes(data: bytes, filename: str) -> Image.Image:
    name = filename.lower()
    if name.endswith((".dcm", ".dicom")):
        return dicom_to_pil(data)
    image = Image.open(io.BytesIO(data))
    return image.convert("RGB")


def dicom_to_pil(data: bytes) -> Image.Image:
    if pydicom is None:
        raise RuntimeError("pydicom is required for DICOM uploads")
    ds = pydicom.dcmread(io.BytesIO(data))
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    window_center = getattr(ds, "WindowCenter", None)
    window_width = getattr(ds, "WindowWidth", None)
    if window_center is not None and window_width is not None:
        center = float(window_center[0] if hasattr(window_center, "__iter__") else window_center)
        width = float(window_width[0] if hasattr(window_width, "__iter__") else window_width)
        low, high = center - width / 2, center + width / 2
        pixels = np.clip(pixels, low, high)
    pixels -= pixels.min()
    max_v = pixels.max() or 1.0
    pixels = (pixels / max_v * 255.0).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def pil_to_numpy(image: Image.Image, size: int = 224) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    return arr
