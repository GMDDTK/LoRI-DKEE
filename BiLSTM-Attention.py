# -*- coding: utf-8 -*-
"""
BiLSTM + Attention for text classification.
数据格式：每行 "text\\tlabel\\tpos"（只用前两列）
支持：GPU、混合精度(--fp16)、类别不平衡权重、可选 GloVe。
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
            if text and label != "":  # 确保文本和标签非空
                X_tok.append(simple_tokenize(text))
                y.append(label)
    return X_tok, y


def build_vocab(tokenized_texts: List[List[str]], min_freq: int = 2, max_size: int = 200000):
    freq = {}
    for toks in tokenized_texts:
        for w in toks:
            freq[w] = freq.get(w, 0) + 1
    items = [(w, c) for w, c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:max_size]
    stoi = {"<pad>": 0, "<unk>": 1}
    for w, _ in items: stoi[w] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def labels_to_ids(labels: List[str]):
    classes = sorted(list(set(labels)), key=lambda x: (len(x), x))
    l2i = {c: i for i, c in enumerate(classes)}
    i2l = {i: c for c, i in l2i.items()}
    y = np.array([l2i[z] for z in labels], dtype=np.int64)
    return y, l2i, i2l


def pad_batch(batch_tok: List[List[str]], stoi: Dict[str, int], max_len: int):
    pad_id, unk_id = 0, 1
    ids = []
    lengths = []
    for toks in batch_tok:
        # NOTE: Removed the check for empty tokens here, as they are filtered out beforehand.
        arr = [stoi.get(w, unk_id) for w in toks][:max_len]
        lengths.append(len(arr))
        if len(arr) < max_len:
            arr = arr + [pad_id] * (max_len - len(arr))
        ids.append(arr)
    return np.array(ids, dtype=np.int64), np.array(lengths, dtype=np.int64)


def compute_class_weights(y_train: np.ndarray, num_classes: int):
    N = len(y_train)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = N / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# -------------------- Embedding 初始化（可选 GloVe） --------------------

def load_glove_to_matrix(glove_path: Path, stoi: Dict[str, int], embed_dim: int):
    vocab_size = len(stoi)
    rng = np.random.default_rng(42)
    mat = rng.normal(0, 0.05, size=(vocab_size, embed_dim)).astype(np.float32)
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

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized, labels, stoi, max_len):
        self.tokenized = tokenized
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        toks = self.tokenized[idx]
        # NOTE: The logic here is simplified. Since empty samples are pre-filtered,
        # we no longer need the check `if len(arr) == 0`.
        arr, length = pad_batch([toks], self.stoi, self.max_len)
        return (
            torch.from_numpy(arr[0]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(length[0], dtype=torch.long)
        )


# -------------------- Model: BiLSTM + Attention --------------------

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H, mask):  # H: [B,T,H], mask: [B,T] (1 for valid, 0 for pad)
        # score = v^T tanh(W_h * H)
        score = self.v(torch.tanh(self.attn(H))).squeeze(-1)  # [B,T]
        score = score.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(score, dim=1)  # [B,T]
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B,H]
        return context, alpha


class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes,
                 pad_idx=0, dropout=0.5, pretrained_emb: np.ndarray = None, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.embedding.weight.requires_grad = not freeze_emb

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        # x: [B,T], lengths: [B]
        emb = self.embedding(x)  # [B,T,E]
        # pack for speed
        lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
        emb_sorted = emb.index_select(0, idx_sort)

        # NOTE: Removed the debug print statement.
        packed = nn.utils.rnn.pack_padded_sequence(emb_sorted, lengths_sorted.cpu(), batch_first=True,
                                                   enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out_sorted, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B,T,H*2]

        # 还原原始顺序
        _, idx_unsort = torch.sort(idx_sort)
        H = out_sorted.index_select(0, idx_unsort)

        # mask: 1 for valid tokens
        max_len = H.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(H.size(0), -1) < lengths.unsqueeze(1)
        context, alpha = self.attn(H, mask)
        h = self.dropout(context)
        logits = self.fc(h)
        return logits, alpha


# -------------------- 训练 / 评估 --------------------

def set_seed(seed: int = 42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb, lb in loader:
            xb = xb.to(device);
            yb = yb.to(device);
            lb = lb.to(device)
            logits, _ = model(xb, lb)
            pred = torch.argmax(logits, dim=-1)
            ys.append(yb.cpu().numpy());
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys);
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"classification2")
    parser.add_argument("--save_dir", default="./ckpt_bilstm_attn")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--vocab_max_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--glove_path", default="glove_trained_300d.txt")
    parser.add_argument("--freeze_emb", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path(args.data_dir)
    Xtr_tok, ytr = read_tsv(data_dir / "train.txt")
    Xva_tok, yva = read_tsv(data_dir / "val.txt")
    Xte_tok, yte = read_tsv(data_dir / "test.txt")

    # +++++++++++++++ FIX STARTS HERE +++++++++++++++
    def filter_empty_samples(tokenized_texts: List[List[str]], labels: List[str]) -> Tuple[List[List[str]], List[str]]:
        """Removes samples where tokenization resulted in an empty list."""
        filtered_X, filtered_y = [], []
        for toks, label in zip(tokenized_texts, labels):
            if toks:  # This checks if the list of tokens is not empty
                filtered_X.append(toks)
                filtered_y.append(label)
        return filtered_X, filtered_y

    print("Filtering samples that became empty after tokenization...")
    Xtr_tok, ytr = filter_empty_samples(Xtr_tok, ytr)
    Xva_tok, yva = filter_empty_samples(Xva_tok, yva)
    Xte_tok, yte = filter_empty_samples(Xte_tok, yte)
    # +++++++++++++++ FIX ENDS HERE +++++++++++++++

    print(f"样本量：train={len(Xtr_tok)}  val={len(Xva_tok)}  test={len(Xte_tok)}")

    stoi, itos = build_vocab(Xtr_tok, min_freq=args.min_freq, max_size=args.vocab_max_size)
    print("Vocab size:", len(stoi))

    y_train, l2i, i2l = labels_to_ids(ytr)
    y_val = np.array([l2i.get(z, -1) for z in yva], dtype=np.int64)
    y_test = np.array([l2i.get(z, -1) for z in yte], dtype=np.int64)

    def filter_pairs(tok_list, y_list):
        keep_tok, keep_y = [], []
        for toks, y in zip(tok_list, y_list):
            if y >= 0:
                keep_tok.append(toks);
                keep_y.append(y)
        return keep_tok, np.array(keep_y, dtype=np.int64)

    Xva_tok, y_val = filter_pairs(Xva_tok, y_val)
    Xte_tok, y_test = filter_pairs(Xte_tok, y_test)

    # 可选加载 GloVe
    pretrained = None
    if args.glove_path:
        glove_file = Path(args.glove_path)
        if glove_file.exists():
            pretrained, found = load_glove_to_matrix(glove_file, stoi, args.embed_dim)
            print(f"GloVe 命中：{found} / {len(stoi)}")
        else:
            print(f"Warning: GloVe path '{args.glove_path}' does not exist. Skipping.")

    # Dataset & Loader（按样本单独pad）
    train_ds = TextDataset(Xtr_tok, y_train, stoi, args.max_len)
    val_ds = TextDataset(Xva_tok, y_val, stoi, args.max_len)
    test_ds = TextDataset(Xte_tok, y_test, stoi, args.max_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(l2i)
    model = BiLSTMAttn(
        vocab_size=len(stoi),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        pad_idx=0,
        dropout=args.dropout,
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
    best_path = os.path.join(args.save_dir, "bilstm_attn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for xb, yb, lb in train_loader:
            xb = xb.to(device);
            yb = yb.to(device);
            lb = lb.to(device)
            optimizer.zero_grad()
            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits, _ = model(xb, lb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(xb, lb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)

        tr_loss = total_loss / len(train_ds)
        # 验证
        model.eval()
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}  val_f1m={val_f1:.4f}  time={time.time() - t0:.1f}s")

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

    # 测试集评估
    if os.path.exists(best_path):
        sd = torch.load(best_path, map_location=device)
        model.load_state_dict(sd["model"])
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"[TEST] acc={test_acc:.4f}  f1-macro={test_f1:.4f}")
    target_names = [i2l[i] for i in range(len(l2i))]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


if __name__ == "__main__":
    main()