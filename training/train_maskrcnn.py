"""Train torchvision Mask R-CNN on COCO-style medical instance annotations.

JSON:
  {"images": [{"id": 1, "file_name": "a.png", "width": 512, "height": 512}],
   "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x,y,w,h], "segmentation": [[...]]}],
   "categories": [{"id": 1, "name": "lesion"}]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


class CocoSegDataset(Dataset):
    def __init__(self, images: Path, coco: dict):
        self.images = images
        self.meta = {im["id"]: im for im in coco["images"]}
        self.by_image = {}
        for ann in coco["annotations"]:
            self.by_image.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.meta)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        info = self.meta[self.ids[idx]]
        img = Image.open(self.images / info["file_name"]).convert("RGB")
        anns = self.by_image.get(info["id"], [])
        boxes, labels, masks = [], [], []
        w_img, h_img = img.size
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            mask = torch.zeros((h_img, w_img), dtype=torch.uint8)
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
            mask[y0:y1, x0:x1] = 1
            masks.append(mask.float())
        if not boxes:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, img.size[1], img.size[0]))
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks_t = torch.stack(masks)
        return F.to_tensor(img), {"boxes": boxes, "labels": labels, "masks": masks_t}


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("weights/maskrcnn_med.pt"))
    args = parser.parse_args()

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    num_classes = 1 + len(coco.get("categories", []))
    ds = CocoSegDataset(args.images, coco)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss)
        print(f"epoch {epoch+1} loss={running/max(1,len(loader)):.4f}")
        torch.save({"model": model.state_dict(), "num_classes": num_classes}, args.out)


if __name__ == "__main__":
    main()
