from __future__ import annotations

import time
import uuid

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import settings
from app.ml.explain import (
    GradCAM,
    cam_to_mask,
    colorize_cam,
    image_to_b64_png,
    mask_to_regions,
    overlay_mask,
)
from app.ml.imaging import CHEST_LABELS, MRI_LABELS
from app.ml.models import load_chest_model, load_mri_model, load_unet
from app.ml.reports import DISCLAIMER, structured_report
from app.schemas import AnalyzeResponse, Finding, Region


IMAGENET_TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MedXpertCAD:
    """Lightweight multi-agent CAD: detection, localization, report, consistency check."""

    def __init__(self) -> None:
        device_name = settings.device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.chest, self.chest_mode = load_chest_model(settings.chest_checkpoint, self.device)
        self.mri, self.mri_mode = load_mri_model(settings.mri_checkpoint, self.device)
        self.unet = load_unet(settings.seg_checkpoint, self.device)

    def analyze_chest(self, image: Image.Image) -> AnalyzeResponse:
        return self._run(
            image=image,
            model=self.chest,
            labels=CHEST_LABELS,
            modality="chest_xray",
            model_mode=self.chest_mode,
            multilabel=True,
            threshold=0.5 if self.chest_mode == "trained" else 0.35,
        )

    def analyze_mri(self, image: Image.Image) -> AnalyzeResponse:
        return self._run(
            image=image,
            model=self.mri,
            labels=MRI_LABELS,
            modality="brain_mri",
            model_mode=self.mri_mode,
            multilabel=False,
            threshold=0.5,
        )

    @torch.inference_mode(False)
    def _run(
        self,
        image: Image.Image,
        model: torch.nn.Module,
        labels: list[str],
        modality: str,
        model_mode: str,
        multilabel: bool,
        threshold: float,
    ) -> AnalyzeResponse:
        started = time.perf_counter()
        rgb = image.convert("RGB")
        batch = IMAGENET_TF(rgb).unsqueeze(0).to(self.device)
        batch.requires_grad_(True)
        cam_engine = GradCAM(model, model.cam_layer())
        try:
            logits = model(batch)
            if multilabel:
                probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            findings = [
                Finding(label=label, probability=float(p), present=bool(p >= threshold))
                for label, p in zip(labels, probs)
            ]
            top_idx = int(np.argmax(probs))
            if multilabel and any(f.present for f in findings):
                top_idx = int(np.argmax([f.probability if f.present else -1 for f in findings]))

            cam = cam_engine(logits, top_idx)
        finally:
            cam_engine.close()

        heatmap = colorize_cam(cam, rgb)
        cam_mask = cam_to_mask(cam, 0.55)
        seg_mask = cam_mask
        dice_proxy = None
        if modality == "brain_mri" and self.unet is not None:
            with torch.no_grad():
                logits_seg = self.unet(batch)
                pred = torch.sigmoid(logits_seg)[0, 0].cpu().numpy()
            seg_mask = (pred >= 0.5).astype(np.uint8)
            dice_proxy = _dice(seg_mask, cam_mask)

        overlay = overlay_mask(rgb, seg_mask)
        regions = [Region(**r) for r in mask_to_regions(seg_mask, labels[top_idx], float(probs[top_idx]))]
        primary = labels[top_idx]
        if multilabel and not any(f.present for f in findings):
            primary = "No finding above threshold"

        critic = "consistent"
        if regions and regions[0].mask_coverage < 0.01 and findings[top_idx].present:
            critic = "low_localization_coverage"

        report = structured_report(
            modality,
            [f.model_dump() for f in findings],
            primary,
            [r.model_dump() for r in regions],
            model_mode,
        )
        latency = (time.perf_counter() - started) * 1000
        return AnalyzeResponse(
            request_id=str(uuid.uuid4()),
            modality=modality,  # type: ignore[arg-type]
            model_mode=model_mode,  # type: ignore[arg-type]
            findings=findings,
            primary_impression=primary,
            heatmap_png_b64=image_to_b64_png(heatmap),
            overlay_png_b64=image_to_b64_png(overlay),
            segmentation_png_b64=image_to_b64_png(_mask_preview(seg_mask)),
            regions=regions,
            report=report,
            dice_proxy=dice_proxy,
            disclaimer=DISCLAIMER,
            latency_ms=round(latency, 1),
            agents={
                "detection": "ChestXrayNet/DenseNet121" if modality == "chest_xray" else "MRIClassifier/ResNet50",
                "localization": "U-Net" if self.unet and modality == "brain_mri" else "Grad-CAM threshold",
                "report": "structured_template",
                "consistency": critic,
            },
        )


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a_b, b_b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    denom = a_b.sum() + b_b.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def _mask_preview(mask: np.ndarray) -> Image.Image:
    scaled = (mask * 255).astype(np.uint8)
    return Image.fromarray(scaled).convert("RGB").resize((224, 224))
