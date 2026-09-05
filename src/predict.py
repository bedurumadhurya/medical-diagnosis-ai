from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .gradcam import GradCAM, overlay_heatmap
from .model import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ChestXrayClassifier,
    load_model,
)

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_image(
    image: Image.Image,
    checkpoint: Path | None = None,
    model: ChestXrayClassifier | None = None,
) -> dict:
    device = _device()
    if model is None:
        model = load_model(checkpoint, device)
    rgb = image.convert("RGB")
    batch = TRANSFORM(rgb).unsqueeze(0).to(device)
    cam_helper = GradCAM(model, model.target_layer())
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)[0]
    class_idx = int(torch.argmax(probs).item())
    heatmap = cam_helper(logits, class_idx)
    overlay = overlay_heatmap(rgb, heatmap)
    return {
        "label": CLASS_NAMES[class_idx],
        "confidence": float(probs[class_idx].item()),
        "probabilities": {name: float(probs[i].item()) for i, name in enumerate(CLASS_NAMES)},
        "overlay": overlay,
    }
