"""Train U-Net on paired MRI / mask folders.

Layout:
  data/raw/mri_seg/images/*.png
  data/raw/mri_seg/masks/*.png   (binary, same filenames)

Reports mean Dice on the validation split (target > 0.95 after sufficient data/epochs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.models import UNet  # noqa: E402


class SegDataset(Dataset):
    def __init__(self, images: Path, masks: Path, names: list[str]):
        self.images = images
        self.masks = masks
        self.names = names
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(self.images / name).convert("RGB")
        mask = Image.open(self.masks / name).convert("L")
        y = self.tf(mask)
        y = (y > 0.5).float()
        return self.tf(img), y


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "mri_unet.pt")
    args = parser.parse_args()

    names = sorted(p.name for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    split = max(1, int(0.85 * len(names)))
    train_ds = SegDataset(args.images, args.masks, names[:split])
    val_ds = SegDataset(args.images, args.masks, names[split:] or names[:1])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
        dice = eval_dice(model, val_loader, device)
        print(f"epoch {epoch+1} val_dice={dice:.4f}")
        if dice >= best:
            best = dice
            torch.save({"model": model.state_dict(), "val_dice": dice}, args.out)
    print(f"saved {args.out} best_dice={best:.4f}")


@torch.no_grad()
def eval_dice(model, loader, device) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        pred = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
        inter = (pred * y).sum()
        denom = pred.sum() + y.sum()
        scores.append(float((2 * inter + 1e-6) / (denom + 1e-6)))
    return float(np.mean(scores)) if scores else 0.0


if __name__ == "__main__":
    main()
