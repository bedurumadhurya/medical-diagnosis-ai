"""Train binary ResNet50 on data/train/{normal,abnormal}."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, ChestXrayClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 for chest X-ray binary classification")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/resnet50_binary.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise SystemExit(
            f"Missing {args.data_dir}. Put images in {args.data_dir}/normal and {args.data_dir}/abnormal."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    class BinaryXrayFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            return ["normal", "abnormal"], {"normal": 0, "abnormal": 1}

    dataset = BinaryXrayFolder(args.data_dir, transform=transform)
    missing = [name for name in ("normal", "abnormal") if not (args.data_dir / name).is_dir()]
    if missing:
        raise SystemExit(f"Expected folders normal/ and abnormal/, missing {missing}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ChestXrayClassifier(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        correct = 0
        n = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / n:.4f}  acc={correct / n:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
