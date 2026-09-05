"""Train CheXNet-style DenseNet121 on NIH ChestX-ray14 (multi-label BCE).

Expected CSV columns: Image Index, Finding Labels (pipe-separated).
Images directory should contain the PNG files referenced by Image Index.

Example:
  python -m training.train_chest --csv data/raw/Data_Entry_2017.csv --images data/raw/images --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.imaging import CHEST_LABELS  # noqa: E402
from app.ml.models import ChestXrayNet  # noqa: E402


class ChestDataset(Dataset):
    def __init__(self, frame, images: Path, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.images = images
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = self.images / row["Image Index"]
        image = Image.open(path).convert("RGB")
        labels = torch.zeros(len(CHEST_LABELS), dtype=torch.float32)
        raw = str(row["Finding Labels"])
        if raw and raw != "No Finding":
            for token in raw.split("|"):
                token = token.strip()
                if token in CHEST_LABELS:
                    labels[CHEST_LABELS.index(token)] = 1.0
        return self.tf(image), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "chest_densenet121.pt")
    parser.add_argument("--limit", type=int, default=0, help="Optional row cap for smoke training")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    df = df.sample(frac=1.0, random_state=42)
    split = int(0.9 * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    train_loader = DataLoader(ChestDataset(train_df, args.images, True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ChestDataset(val_df, args.images, False), batch_size=args.batch_size, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXrayNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
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
        auc = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1} loss={running/len(train_df):.4f} mean_auc={auc:.4f}")
        if auc >= best:
            best = auc
            torch.save({"model": model.state_dict(), "labels": CHEST_LABELS, "val_auc": auc}, args.out)
    print(f"saved {args.out} best_mean_auc={best:.4f}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu()
        ys.append(y)
        ps.append(p)
    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(ps).numpy()
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].min() == y_true[:, i].max():
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(sum(aucs) / len(aucs)) if aucs else 0.0


if __name__ == "__main__":
    main()
