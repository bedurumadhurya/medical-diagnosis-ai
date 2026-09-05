"""Grad-CAM heatmap over the last ResNet conv block."""

import numpy as np
import torch
from PIL import Image
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, logits: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0
    color = np.zeros((cam_arr.shape[0], cam_arr.shape[1], 3), dtype=np.float32)
    color[..., 0] = cam_arr
    color[..., 1] = cam_arr * 0.3
    base = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mixed = np.clip((1 - alpha) * base + alpha * color, 0, 1)
    return Image.fromarray(np.uint8(mixed * 255))
