"""Train ResNet50 on a Kaggle-style Brain Tumor MRI folder layout:

  data/raw/brain_mri/{Training,Testing}/{glioma,meningioma,notumor,pituitary}/*.jpg

Example:
  python -m training.train_mri --data data/raw/brain_mri --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.imaging import MRI_LABELS  # noqa: E402
from app.ml.models import MRIClassifier  # noqa: E402


class FolderDataset(Dataset):
    def __init__(self, root: Path, train: bool):
        self.items = []
        for label in MRI_LABELS:
            folder = root / label
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.items.append((path, MRI_LABELS.index(label)))
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.tf(image), y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "mri_resnet50.pt")
    args = parser.parse_args()

    train_root = args.data / "Training" if (args.data / "Training").exists() else args.data
    val_root = args.data / "Testing" if (args.data / "Testing").exists() else args.data
    train_ds, val_ds = FolderDataset(train_root, True), FolderDataset(val_root, False)
    if len(train_ds) == 0:
        raise SystemExit(f"No images found under {train_root}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRIClassifier(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        acc, report = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1} loss={running/len(train_ds):.4f} acc={acc:.4f}")
        print(report)
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "labels": MRI_LABELS, "val_acc": acc}, args.out)
    print(f"saved {args.out} best_acc={best:.4f}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().tolist()
        yp.extend(pred)
        yt.extend(y.tolist())
    return accuracy_score(yt, yp), classification_report(yt, yp, target_names=MRI_LABELS, zero_division=0)


if __name__ == "__main__":
    main()
