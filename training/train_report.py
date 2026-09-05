"""Train a label-conditioned LSTM decoder and report corpus BLEU-4.

Input JSONL lines:
  {"labels": {"Pneumonia": 1, "Effusion": 1}, "report": "Findings include ..."}

Optional BioWordVec: --biowordvec path/to/BioWordVec_PubMed_MIMICIII_d200.bin
(gensim KeyedVectors). If omitted, embeddings are learned from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.imaging import CHEST_LABELS  # noqa: E402
from app.ml.models import ReportDecoder  # noqa: E402

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.replace("\n", " ").split() if t]


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu4(candidate: list[str], reference: list[str]) -> float:
    if not candidate:
        return 0.0
    precisions = []
    for n in range(1, 5):
        c, r = ngrams(candidate, n), ngrams(reference, n)
        overlap = sum(min(v, r[k]) for k, v in c.items())
        total = max(1, sum(c.values()))
        precisions.append(overlap / total)
    if min(precisions) == 0:
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions) / 4)
    bp = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(1, len(candidate)))
    return bp * geo


class ReportDataset(Dataset):
    def __init__(self, rows, stoi, max_len=80):
        self.rows = rows
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def _ids(self, text):
        ids = [BOS] + [self.stoi.get(t, UNK) for t in tokenize(text)][: self.max_len - 2] + [EOS]
        ids += [PAD] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        row = self.rows[idx]
        vec = torch.zeros(len(CHEST_LABELS))
        for k, v in row.get("labels", {}).items():
            if k in CHEST_LABELS:
                vec[CHEST_LABELS.index(k)] = float(v)
        ids = torch.tensor(self._ids(row["report"]), dtype=torch.long)
        return vec, ids[:-1], ids[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "report_decoder.pt")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    for row in rows:
        for tok in tokenize(row["report"]):
            if tok not in vocab:
                vocab.append(tok)
    stoi = {t: i for i, t in enumerate(vocab)}
    split = max(1, int(0.9 * len(rows)))
    train_ds = ReportDataset(rows[:split], stoi)
    val_rows = rows[split:] or rows[:1]
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReportDecoder(vocab_size=len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for labels, inp, tgt in loader:
            labels, inp, tgt = labels.to(device), inp.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(labels, inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item()
        score = eval_bleu(model, val_rows, stoi, vocab, device)
        print(f"epoch {epoch+1} loss={total/len(loader):.4f} bleu4={score:.4f}")
        torch.save({"model": model.state_dict(), "vocab": vocab, "bleu4": score}, args.out)
    print(f"saved {args.out}")


@torch.no_grad()
def eval_bleu(model, rows, stoi, vocab, device, max_len=80) -> float:
    model.eval()
    scores = []
    itos = {i: t for t, i in stoi.items()}
    for row in rows:
        vec = torch.zeros(1, len(CHEST_LABELS), device=device)
        for k, v in row.get("labels", {}).items():
            if k in CHEST_LABELS:
                vec[0, CHEST_LABELS.index(k)] = float(v)
        tokens = [BOS]
        for _ in range(max_len - 1):
            inp = torch.tensor([tokens], device=device)
            logits = model(vec, inp)
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS:
                break
            tokens.append(nxt)
        cand = [itos.get(i, "") for i in tokens if i not in {PAD, BOS, EOS}]
        scores.append(bleu4(cand, tokenize(row["report"])))
    return sum(scores) / len(scores)


if __name__ == "__main__":
    main()
