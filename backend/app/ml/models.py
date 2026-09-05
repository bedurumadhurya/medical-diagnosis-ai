from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, ResNet50_Weights, densenet121, resnet50

from app.ml.imaging import CHEST_LABELS, MRI_LABELS


class ChestXrayNet(nn.Module):
    """CheXNet-style DenseNet121 for 14 NIH ChestX-ray pathologies."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def cam_layer(self) -> nn.Module:
        return self.backbone.features.denseblock4


class MRIClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def cam_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """U-Net for MRI tumor / abnormality segmentation (Dice target > 0.95 after training)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class ReportDecoder(nn.Module):
    """Image-encoder + LSTM decoder for radiology-style sentences (BLEU-4 eval in training/)."""

    def __init__(self, vocab_size: int, embed_dim: int = 200, hidden: int = 512, num_labels: int = 14):
        super().__init__()
        self.label_proj = nn.Linear(num_labels, hidden)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, label_probs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        h0 = torch.tanh(self.label_proj(label_probs)).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        emb = self.embed(tokens)
        out, _ = self.lstm(emb, (h0, c0))
        return self.head(out)


def load_chest_model(checkpoint, device: torch.device) -> tuple[ChestXrayNet, str]:
    model = ChestXrayNet(num_classes=len(CHEST_LABELS), pretrained=True)
    mode = "imagenet_demo"
    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        mode = "trained"
    model.to(device).eval()
    return model, mode


def load_mri_model(checkpoint, device: torch.device) -> tuple[MRIClassifier, str]:
    model = MRIClassifier(num_classes=len(MRI_LABELS), pretrained=True)
    mode = "imagenet_demo"
    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        mode = "trained"
    model.to(device).eval()
    return model, mode


def load_unet(checkpoint, device: torch.device) -> UNet | None:
    if not checkpoint or not checkpoint.exists():
        return None
    model = UNet()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return model.to(device).eval()
