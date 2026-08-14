# -*- coding: utf-8 -*-
"""
Shared vocab + dataset code, so RNN/GRU/LSTM runs see identical inputs.

Character-level, not word-level: the classes being predicted (hamza_alef,
ta_marbuta, alef_maqsura, rasm_dots, ...) are defined by which single
character differs inside a word. A word-level vocab hides that signal
behind an opaque token id and can only recognize a misspelling it already
saw verbatim in training; character-level lets the model see the actual
orthographic difference and generalize to unseen words.
"""
from __future__ import annotations

import os
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def read_tsv(split):
    rows = []
    with open(os.path.join(DATA_DIR, f"{split}.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text, label = line.rsplit("\t", 1)
            rows.append((text, label))
    return rows


def read_labels():
    with open(os.path.join(DATA_DIR, "labels.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def build_vocab(rows, min_freq=2):
    counts = Counter()
    for text, _ in rows:
        counts.update(list(text))
    itos = [PAD, UNK] + [w for w, c in counts.items() if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


class ClassificationDataset(Dataset):
    def __init__(self, rows, stoi, label2idx):
        self.rows = rows
        self.stoi = stoi
        self.label2idx = label2idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        ids = [self.stoi.get(c, self.stoi[UNK]) for c in text]
        if not ids:
            ids = [self.stoi[UNK]]
        return torch.tensor(ids, dtype=torch.long), self.label2idx[label], text


def collate(batch):
    seqs, labels, texts = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    maxlen = int(lengths.max())
    padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels, texts
