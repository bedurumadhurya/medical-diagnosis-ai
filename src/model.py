"""ResNet50 binary classifier: normal vs abnormal chest X-ray."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("normal", "abnormal")


class ChestXrayClassifier(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


def load_model(checkpoint: Path | None, device: torch.device) -> ChestXrayClassifier:
    pretrained = checkpoint is None or not checkpoint.exists()
    model = ChestXrayClassifier(pretrained=pretrained)
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
