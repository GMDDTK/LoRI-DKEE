# -*- coding: utf-8 -*-
"""
BiGRU + Attention 文本分类
- 支持加载 GloVe
- 支持 freeze / finetune embedding
- 训练、验证、早停与保存
"""

import os
import re
import time
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------- 数据处理 ----------------

def simple_tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def read_tsv(path: Path):
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                texts.append(simple_tokenize(parts[0]))
                labels.append(parts[1].strip())
    return texts, labels


def build_vocab(token_lists, min_freq=2, max_size=200000):
    from collections import Counter
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counter.most_common(max_size):
        if c >= min_freq:
            vocab[w] = len(vocab)
    itos = {i: w for w, i in vocab.items()}
    return vocab, itos


def labels_to_ids(labels):
    uniq = sorted(set(labels))
    l2i = {l: i for i, l in enumerate(uniq)}
    i2l = {i: l for l, i in l2i.items()}
    return np.array([l2i[l] for l in labels], dtype=np.int64), l2i, i2l


def encode_tokens(tokens, stoi, max_len=256):
    ids = [stoi.get(t, stoi["<UNK>"]) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [stoi["<PAD>"]] * (max_len - len(ids))
    return ids, len(tokens[:max_len])


class TextDataset(Dataset):
    def __init__(self, token_lists, labels, stoi, max_len=256):
        self.data = [encode_tokens(toks, stoi, max_len) for toks in token_lists]
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        ids, l = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(l, dtype=torch.long)


# ---------------- GloVe ----------------

def load_glove_to_matrix(glove_path: Path, stoi: dict, embed_dim: int):
    matrix = np.random.uniform(-0.05, 0.05, (len(stoi), embed_dim))
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embed_dim + 1:
                continue
            w, vec = parts[0], np.array(parts[1:], dtype=np.float32)
            if w in stoi:
                matrix[stoi[w]] = vec
                found += 1
    return matrix, found


# ---------------- 模型 ----------------

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, H, mask=None):
        # H: [B, L, 2H]
        scores = self.attn(H).squeeze(-1)  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=-1)  # [B, L]
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B, 2H]
        return context, alpha


class BiGRUAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 num_classes, pad_idx=0, dropout=0.5,
                 pretrained_emb=None, freeze_emb=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_emb))
        if freeze_emb:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=num_layers,
                          bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        emb = self.embedding(x)  # [B, L, D]
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        context, alpha = self.attn(out)
        context = self.dropout(context)
        return self.fc(context), alpha


# ---------------- 工具 ----------------

def compute_class_weights(y, num_classes):
    N = len(y)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = N / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb, lb in loader:
        xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
        logits, _ = model(xb, lb)
        pred = torch.argmax(logits, dim=-1)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m, y_true, y_pred


# ---------------- 主函数 ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"classification2")
    parser.add_argument("--save_dir", default="./ckpt_bigru_attn")
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

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path(args.data_dir)
    Xtr_tok, ytr = read_tsv(data_dir / "train.txt")
    Xva_tok, yva = read_tsv(data_dir / "val.txt")
    Xte_tok, yte = read_tsv(data_dir / "test.txt")
    print(f"样本量：train={len(Xtr_tok)}  val={len(Xva_tok)}  test={len(Xte_tok)}")

    stoi, itos = build_vocab(Xtr_tok, min_freq=args.min_freq, max_size=args.vocab_max_size)
    print("Vocab size:", len(stoi))

    y_train, l2i, i2l = labels_to_ids(ytr)
    y_val  = np.array([l2i.get(z, -1) for z in yva], dtype=np.int64)
    y_test = np.array([l2i.get(z, -1) for z in yte], dtype=np.int64)
    def filter_pairs(tok_list, y_list):
        keep_tok, keep_y = [], []
        for toks, y in zip(tok_list, y_list):
            if y >= 0:
                keep_tok.append(toks); keep_y.append(y)
        return keep_tok, np.array(keep_y, dtype=np.int64)
    Xva_tok, y_val = filter_pairs(Xva_tok, y_val)
    Xte_tok, y_test = filter_pairs(Xte_tok, y_test)

    # 可选加载 GloVe
    pretrained = None
    if args.glove_path:
        pretrained, found = load_glove_to_matrix(Path(args.glove_path), stoi, args.embed_dim)
        print(f"GloVe 命中：{found} / {len(stoi)}")

    # Dataset & Loader
    train_ds = TextDataset(Xtr_tok, y_train, stoi, args.max_len)
    val_ds   = TextDataset(Xva_tok, y_val, stoi, args.max_len)
    test_ds  = TextDataset(Xte_tok, y_test, stoi, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    num_classes = len(l2i)
    model = BiGRUAttn(
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
    best_path = os.path.join(args.save_dir, "bigru_attn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
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
        model.eval()
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

    # 测试
    model.load_state_dict(torch.load(best_path)["model"])
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"[TEST] acc={test_acc:.4f} | f1-macro={test_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=list(l2i.keys()), digits=4))


if __name__ == "__main__":
    main()
