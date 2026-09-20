# -*- coding: utf-8 -*-
"""
TextCNN（Kim CNN） for sentence/document classification.
数据格式：每行 "text\\tlabel\\tpos"（只用前两列）
支持：GPU、混合精度(--fp16)、类别权重、可选 GloVe 词向量加载/冻结。
PyTorch 1.x + Python 3.7 兼容
"""

import os
import re
import json
import math
import time
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report, accuracy_score, f1_score

# -------------------- 数据读取与预处理 --------------------

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def read_tsv(path: Path) -> Tuple[List[List[str]], List[str]]:
    X_tok, y = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            text = parts[0].strip()
            label = parts[1].strip()
            if text and label != "":
                X_tok.append(simple_tokenize(text))
                y.append(label)
    return X_tok, y

def build_vocab(tokenized_texts: List[List[str]], min_freq: int = 2, max_size: int = 200000):
    freq = {}
    for toks in tokenized_texts:
        for w in toks:
            freq[w] = freq.get(w, 0) + 1
    # 排除低频
    items = [(w, c) for w, c in freq.items() if c >= min_freq]
    # 频次排序
    items.sort(key=lambda x: (-x[1], x[0]))
    # 限制大小
    items = items[:max_size]
    # 预留特殊符号
    stoi = {"<pad>": 0, "<unk>": 1}
    for w, _ in items:
        stoi[w] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def texts_to_ids(tokenized_texts: List[List[str]], stoi: Dict[str, int], max_len: int):
    pad_id, unk_id = 0, 1
    ids = []
    lengths = []
    for toks in tokenized_texts:
        arr = [stoi.get(w, unk_id) for w in toks][:max_len]
        lengths.append(len(arr))
        if len(arr) < max_len:
            arr = arr + [pad_id] * (max_len - len(arr))
        ids.append(arr)
    return np.array(ids, dtype=np.int64), np.array(lengths, dtype=np.int64)

def labels_to_ids(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    classes = sorted(list(set(labels)), key=lambda x: (len(x), x))
    l2i = {c: i for i, c in enumerate(classes)}
    i2l = {i: c for c, i in l2i.items()}
    y = np.array([l2i[z] for z in labels], dtype=np.int64)
    return y, l2i, i2l

def compute_class_weights(y_train: np.ndarray, num_classes: int):
    # 反频率权重：w_c = N / (num_classes * n_c)
    N = len(y_train)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = N / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

# -------------------- Embedding 初始化（可选 GloVe） --------------------

def load_glove_to_matrix(glove_path: Path, stoi: Dict[str, int], embed_dim: int):
    """
    从 GloVe 文本向量加载（例如 glove.6B.300d.txt），只填充词表中有的词；其余随机。
    返回：np.ndarray [vocab_size, embed_dim]
    """
    vocab_size = len(stoi)
    rng = np.random.default_rng(42)
    mat = rng.normal(0, 0.05, size=(vocab_size, embed_dim)).astype(np.float32)
    # pad=0 向量置零
    mat[0] = 0.0

    found = 0
    if glove_path and glove_path.exists():
        with glove_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n ").split(" ")
                if len(parts) < embed_dim + 1:
                    continue
                w = parts[0]
                if w in stoi:
                    try:
                        vec = np.asarray(parts[1:1 + embed_dim], dtype=np.float32)
                        if vec.shape[0] != embed_dim:
                            continue
                        mat[stoi[w]] = vec
                        found += 1
                    except Exception:
                        continue
    return mat, found

# -------------------- Dataset / DataLoader --------------------

class NpDataset(torch.utils.data.Dataset):
    def __init__(self, X_ids: np.ndarray, y: np.ndarray, lengths: np.ndarray):
        self.X_ids = X_ids
        self.y = y
        self.lengths = lengths
    def __len__(self):
        return self.X_ids.shape[0]
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_ids[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long)
        )

# -------------------- Model: TextCNN --------------------

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes=(3,4,5), num_filters=100,
                 dropout=0.5, pad_idx=0, pretrained_emb: np.ndarray = None, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.embedding.weight.requires_grad = not freeze_emb

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):  # x: [B, T]
        emb = self.embedding(x)          # [B, T, E]
        emb = emb.transpose(1, 2)        # [B, E, T]
        conv_outs = []
        for conv in self.convs:
            y = F.relu(conv(emb))        # [B, F, T']
            y = F.max_pool1d(y, kernel_size=y.shape[-1]).squeeze(-1)  # [B, F]
            conv_outs.append(y)
        h = torch.cat(conv_outs, dim=1)  # [B, F * K]
        h = self.dropout(h)
        logits = self.fc(h)              # [B, C]
        return logits

# -------------------- 训练 / 评估 --------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=-1)
            ys.append(yb.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m, y_true, y_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"classification2")
    parser.add_argument("--save_dir", default="./ckpt_textcnn")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--vocab_max_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--kernel_sizes", default="3,4,5")
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--glove_path", default="glove_trained_300d.txt", help="例如 glove_trained_300d.txt")
    parser.add_argument("--freeze_emb", action="store_true")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path(args.data_dir)
    Xtr_tok, ytr = read_tsv(data_dir / "train.txt")
    Xva_tok, yva = read_tsv(data_dir / "val.txt")
    Xte_tok, yte = read_tsv(data_dir / "test.txt")
    print(f"样本量：train={len(Xtr_tok)}  val={len(Xva_tok)}  test={len(Xte_tok)}")

    stoi, itos = build_vocab(Xtr_tok, min_freq=args.min_freq, max_size=args.vocab_max_size)
    print("Vocab size:", len(stoi))

    Xtr_ids, Ltr = texts_to_ids(Xtr_tok, stoi, args.max_len)
    Xva_ids, Lva = texts_to_ids(Xva_tok, stoi, args.max_len)
    Xte_ids, Lte = texts_to_ids(Xte_tok, stoi, args.max_len)
    y_train, l2i, i2l = labels_to_ids(ytr)
    y_val  = np.array([l2i.get(z, -1) for z in yva], dtype=np.int64)
    y_test = np.array([l2i.get(z, -1) for z in yte], dtype=np.int64)

    # 过滤掉 val/test 中不可识别标签（极少见）
    def filter_invalid(X_ids, y, L):
        ok = y >= 0
        return X_ids[ok], y[ok], L[ok]
    Xva_ids, y_val, Lva = filter_invalid(Xva_ids, y_val, Lva)
    Xte_ids, y_test, Lte = filter_invalid(Xte_ids, y_test, Lte)

    # 可选加载 GloVe
    pretrained = None
    found = 0
    if args.glove_path:
        pretrained, found = load_glove_to_matrix(Path(args.glove_path), stoi, args.embed_dim)
        print(f"GloVe 命中：{found} / {len(stoi)}")

    train_ds = NpDataset(Xtr_ids, y_train, Ltr)
    val_ds   = NpDataset(Xva_ids, y_val, Lva)
    test_ds  = NpDataset(Xte_ids, y_test, Lte)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(l2i)
    model = TextCNN(
        vocab_size=len(stoi),
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        kernel_sizes=tuple(int(k) for k in args.kernel_sizes.split(",")),
        num_filters=args.num_filters,
        dropout=args.dropout,
        pad_idx=0,
        pretrained_emb=pretrained,
        freeze_emb=args.freeze_emb
    ).to(device)

    class_weights = compute_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_fp16 = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = -1.0
    bad_epochs = 0
    best_path = os.path.join(args.save_dir, "textcnn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for xb, yb, _ in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)

        tr_loss = total_loss / len(train_ds)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}  val_f1m={val_f1:.4f}  time={time.time()-t0:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save({"model": model.state_dict(),
                        "stoi": stoi, "itos": {int(k): v for k, v in itos.items()},
                        "label_map": l2i, "args": vars(args)}, best_path)
            print(f"  ✓ 新最佳，已保存到 {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("早停触发。")
                break

    # 测试集评估（加载最佳）
    if os.path.exists(best_path):
        sd = torch.load(best_path, map_location=device)
        model.load_state_dict(sd["model"])
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"[TEST] acc={test_acc:.4f}  f1-macro={test_f1:.4f}")
    # 打印详细报告（标签按原字符串次序）
    target_names = [i2l[i] for i in range(len(l2i))]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()
