# -*- coding: utf-8 -*-
"""
Linear SVM + TF-IDF（word + char）；可选拼接 Word2Vec（自训或加载预训练）
数据：每行 "text\\tlabel\\tpos"（pos忽略）
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

# ============ 可选：Word2Vec ============
try:
    from gensim.models import Word2Vec, KeyedVectors
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False
# =======================================

BASE_DIR = r"classification2"

def read_tsv(path: Path):
    X, y = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            text = parts[0].strip()
            label = parts[1].strip()
            if text and label != "":
                X.append(text)
                y.append(label)
    return X, np.array(y)

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

def simple_tokenize(text):
    return TOKEN_RE.findall(text.lower())

class W2VTransformer(BaseEstimator, TransformerMixin):
    """
    将文本转换为句向量（平均或TF-IDF加权平均）：
    - mode = 'none'：禁用
    - mode = 'train'：在训练集上自训 Word2Vec
    - mode = 'load'：加载预训练向量（bin/kv）
    返回 csr_matrix[n_samples, dim]
    """
    def __init__(self, mode='none', vector_size=300, w2v_path=None, tfidf_weight=True, min_count=2, window=5, sg=1, workers=4, seed=42):
        self.mode = mode
        self.vector_size = vector_size
        self.w2v_path = w2v_path
        self.tfidf_weight = tfidf_weight
        self.min_count = min_count
        self.window = window
        self.sg = sg
        self.workers = workers
        self.seed = seed
        # 训练时内部拟合一个 word-level TF-IDF 仅用于权重（不影响主TF-IDF特征）
        self._idf = {}
        self._dim = vector_size
        self._w2v = None
        self._trained_vocab = False

    def fit(self, X, y=None):
        if self.mode == 'none':
            self._dim = self.vector_size
            return self
        if not HAS_GENSIM:
            raise RuntimeError("需要 gensim（pip install gensim==4.2.0）")

        tokenized = [simple_tokenize(t) for t in X]

        if self.mode == 'train':
            self._w2v = Word2Vec(sentences=tokenized, vector_size=self.vector_size,
                                 min_count=self.min_count, window=self.window,
                                 sg=self.sg, workers=self.workers, seed=self.seed)
            self._dim = self.vector_size
            self._trained_vocab = True
        elif self.mode == 'load':
            if not self.w2v_path or not Path(self.w2v_path).exists():
                raise FileNotFoundError("w2v_path 无效或不存在")
            # 自动识别是否为二进制
            if str(self.w2v_path).lower().endswith(".bin"):
                self._w2v = KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)
            else:
                try:
                    self._w2v = KeyedVectors.load(self.w2v_path)
                except Exception:
                    # 尝试以文本格式读取
                    self._w2v = KeyedVectors.load_word2vec_format(self.w2v_path, binary=False)
            self._dim = self._w2v.vector_size
            self._trained_vocab = False
        else:
            raise ValueError("mode 必须为 none/train/load")

        if self.tfidf_weight:
            # 仅统计IDF（简易实现）
            df = {}
            N = len(tokenized)
            for toks in tokenized:
                for w in set(toks):
                    df[w] = df.get(w, 0) + 1
            import math
            self._idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        return self

    def transform(self, X):
        if self.mode == 'none' or self._w2v is None:
            # 返回零向量占位
            mat = np.zeros((len(X), self._dim), dtype=np.float32)
            return csr_matrix(mat)
        rows = []
        for text in X:
            toks = simple_tokenize(text)
            vecs = []
            weights = []
            for w in toks:
                if w in self._w2v:
                    vecs.append(self._w2v[w])
                    weights.append(self._idf.get(w, 1.0) if self.tfidf_weight else 1.0)
            if vecs:
                v = np.average(np.stack(vecs, axis=0), axis=0, weights=np.array(weights))
            else:
                v = np.zeros((self._dim,), dtype=np.float32)
            rows.append(v)
        mat = np.vstack(rows).astype(np.float32)
        return csr_matrix(mat)  # 转稀疏以便与 TF-IDF 拼接

def build_features(include_w2v=False, w2v_kwargs=None):
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        token_pattern=r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?",
        lowercase=True,
        strip_accents="unicode"
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        lowercase=True
    )
    feats = [("word_tfidf", word_tfidf), ("char_tfidf", char_tfidf)]
    if include_w2v:
        feats.append(("w2v", W2VTransformer(**(w2v_kwargs or {}))))
    return FeatureUnion(feats)

def train_eval(model, Xtr, ytr, Xva, yva, Xte, yte, desc="LinearSVC"):
    model.fit(Xtr, ytr)
    for split, X, y in [("VAL", Xva, yva), ("TEST", Xte, yte)]:
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        f1m = f1_score(y, pred, average="macro")
        print(f"[{desc}] {split} Acc={acc:.4f}  F1-macro={f1m:.4f}")
        print(classification_report(y, pred, digits=4))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=BASE_DIR)
    parser.add_argument("--use_w2v", action="store_true", help="拼接 Word2Vec 句向量")
    parser.add_argument("--w2v_mode", default="train", choices=["none", "train", "load"])
    parser.add_argument("--w2v_path", default=None, help="预训练向量路径(.bin/.kv/.txt)，w2v_mode=load时必填")
    parser.add_argument("--w2v_dim", type=int, default=300)
    parser.add_argument("--w2v_tfidf_weight", action="store_true", help="Word2Vec 采用TF-IDF加权平均")
    args = parser.parse_args()

    train_p = Path(args.data_dir) / "train.txt"
    val_p   = Path(args.data_dir) / "val.txt"
    test_p  = Path(args.data_dir) / "test.txt"
    Xtr, ytr = read_tsv(train_p)
    Xva, yva = read_tsv(val_p)
    Xte, yte = read_tsv(test_p)

    print(f"样本量：train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    feats = build_features(
        include_w2v=args.use_w2v,
        w2v_kwargs=dict(
            mode=args.w2v_mode if args.use_w2v else "none",
            vector_size=args.w2v_dim,
            w2v_path=args.w2v_path,
            tfidf_weight=args.w2v_tfidf_weight
        )
    )
    clf = LinearSVC(class_weight="balanced")  # 文献数据常不平衡
    pipe = Pipeline([("feats", feats), ("clf", clf)])
    train_eval(pipe, Xtr, ytr, Xva, yva, Xte, yte, desc="LinearSVC")


if __name__ == "__main__":
    main()
