# -*- coding: utf-8 -*-
"""
BERT 微调用于文本分类（支持多/二分类）
数据：每行 "text\\tlabel\\tpos"（仅用前两列）
依赖：transformers==4.27.4, torch, sklearn, numpy, tqdm
"""

import os
import math
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

# ---------------- I/O ----------------

def read_tsv(path: Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            text = parts[0].strip()
            label = parts[1].strip()
            if text and label != "":
                texts.append(text)
                labels.append(label)
    return texts, labels

def build_label_map(train_labels: List[str]):
    classes = sorted(list(set(train_labels)), key=lambda x: (len(x), x))
    l2i = {c: i for i, c in enumerate(classes)}
    i2l = {i: c for c, i in l2i.items()}
    return l2i, i2l

def map_labels(labels: List[str], l2i):
    arr = np.array([l2i.get(x, -1) for x in labels], dtype=np.int64)
    mask = arr >= 0
    return arr, mask

# --------------- Dataset / Collate ---------------

class TsvDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], int(self.labels[idx])

def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc["labels"] = labels
        return enc
    return collate

# --------------- Utils ---------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_class_weights(y: np.ndarray, num_classes: int):
    N = len(y)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = N / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        pred = torch.argmax(logits, dim=-1)
        ys.append(labels.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m, y_true, y_pred

# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"classification2")
    parser.add_argument("--model_name_or_path", default="G:\\pretrain_models\\bert-base-uncased",
                        help="可填 bert-base-uncased 或本地目录，如 F:\\models\\bert-base-uncased")
    parser.add_argument("--save_dir", default="./ckpt_bert")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=2, help="early stopping patience")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path(args.data_dir)
    Xtr, ytr = read_tsv(data_dir / "train.txt")
    Xva, yva = read_tsv(data_dir / "val.txt")
    Xte, yte = read_tsv(data_dir / "test.txt")
    print(f"样本量：train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    # 标签映射（按训练集确定）
    l2i, i2l = build_label_map(ytr)
    ytr_id, _ = map_labels(ytr, l2i)
    yva_id, mva = map_labels(yva, l2i)
    yte_id, mte = map_labels(yte, l2i)

    # 过滤 val/test 中不可识别标签（通常不会发生）
    Xva = [t for t, ok in zip(Xva, mva) if ok]; yva_id = yva_id[mva]
    Xte = [t for t, ok in zip(Xte, mte) if ok]; yte_id = yte_id[mte]

    # 构建数据集
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_ds = TsvDataset(Xtr, ytr_id)
    val_ds   = TsvDataset(Xva, yva_id)
    test_ds  = TsvDataset(Xte, yte_id)
    collate_fn = make_collate_fn(tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    num_classes = len(l2i)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_classes
    ).to(device)

    # 类别不平衡权重
    class_weights = compute_class_weights(ytr_id, num_classes).to(device)

    # 优化器与调度
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_fp16 = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = -1.0
    bad_epochs = 0
    best_dir = os.path.join(args.save_dir, "bert_best")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), 1):
            labels = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits = model(**batch).logits
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
                scaler.scale(loss / args.grad_accum).backward()
                if step % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                logits = model(**batch).logits
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                (loss / args.grad_accum).backward()
                if step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            total_loss += loss.item()

        # 验证
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={(total_loss/len(train_loader)):.4f} | val_acc={val_acc:.4f} | val_f1m={val_f1:.4f} | time={time.time()-t0:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            # 保存可再次 from_pretrained 的目录
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            # 也保存标签映射
            with open(os.path.join(best_dir, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({"l2i": l2i, "i2l": i2l}, f, ensure_ascii=False)
            print(f"  ✓ 新最佳，已保存到 {best_dir}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("早停触发。")
                break

    # 测试评估（加载最佳权重）
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    names = [v for k, v in sorted(i2l.items(), key=lambda x: x[0])]
    print(f"[TEST] acc={test_acc:.4f} | f1-macro={test_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))


if __name__ == "__main__":
    main()
