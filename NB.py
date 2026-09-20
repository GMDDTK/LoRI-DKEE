# -*- coding: utf-8 -*-
"""
Complement Naive Bayes + TF-IDF（word + char）
注意：NB 要求非负特征，因此不建议拼接含负值的 Word2Vec。
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path

from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score

BASE_DIR = r"classification2"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

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

def build_features():
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.9,
        token_pattern=r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?",
        lowercase=True, strip_accents="unicode"
    )
    char_tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, lowercase=True)
    return FeatureUnion([("word_tfidf", word_tfidf), ("char_tfidf", char_tfidf)])

def train_eval(model, Xtr, ytr, Xva, yva, Xte, yte):
    model.fit(Xtr, ytr)
    for name, X, y in [("VAL", Xva, yva), ("TEST", Xte, yte)]:
        pred = model.predict(X)
        acc = accuracy_score(y, pred); f1m = f1_score(y, pred, average="macro")
        print(f"[CNB] {name} Acc={acc:.4f}  F1-macro={f1m:.4f}")
        print(classification_report(y, pred, digits=4))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=BASE_DIR)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    train_p = Path(args.data_dir) / "train.txt"
    val_p   = Path(args.data_dir) / "val.txt"
    test_p  = Path(args.data_dir) / "test.txt"
    Xtr, ytr = read_tsv(train_p); Xva, yva = read_tsv(val_p); Xte, yte = read_tsv(test_p)
    print(f"样本量：train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    feats = build_features()
    clf = ComplementNB(alpha=args.alpha)
    pipe = Pipeline([("feats", feats), ("clf", clf)])
    train_eval(pipe, Xtr, ytr, Xva, yva, Xte, yte)

if __name__ == "__main__":
    main()
