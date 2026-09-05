from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, logits: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def colorize_cam(cam: np.ndarray, image: Image.Image, alpha: float = 0.45) -> Image.Image:
    heatmap = (cam * 255).astype(np.uint8)
    heat_img = Image.fromarray(heatmap, mode="L").resize(image.size, Image.Resampling.BILINEAR)
    heat_rgb = Image.fromarray(np.stack(_jet(np.asarray(heat_img) / 255.0), axis=-1))
    base = image.convert("RGB").resize(heat_rgb.size)
    return Image.blend(base, heat_rgb, alpha)


def cam_to_mask(cam: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    return (cam >= threshold).astype(np.uint8)


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 64, 96)) -> Image.Image:
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    m = (np.asarray(mask_img) > 127)[..., None].astype(np.float32)
    tint = np.zeros_like(base)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    blended = np.clip(base * (1 - 0.43 * m) + tint * (0.43 * m), 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    import base64

    return base64.b64encode(buf.getvalue()).decode("ascii")


def mask_to_regions(mask: np.ndarray, label: str, score: float) -> list[dict]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    coverage = float(mask.mean())
    return [
        {
            "label": label,
            "score": score,
            "bbox": [x0 / mask.shape[1], y0 / mask.shape[0], x1 / mask.shape[1], y1 / mask.shape[0]],
            "mask_coverage": coverage,
        }
    ]


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    inter = np.logical_and(pred_b, target_b).sum()
    denom = pred_b.sum() + target_b.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def _jet(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)
