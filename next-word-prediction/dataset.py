# -*- coding: utf-8 -*-
"""Shared vocab and windowing code for all 15 (architecture, length) runs."""
from __future__ import annotations

import os
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def read_tokens(split):
    with open(os.path.join(DATA_DIR, f"{split}.txt"), encoding="utf-8") as f:
        return f.read().split()


def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    itos = [PAD, UNK] + [w for w, c in counts.items() if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def stride_for_target_count(n_tokens, seq_len, target_n):
    """
    Pick a window stride so a dataset ends up with about `target_n`
    examples, regardless of seq_len. Needed because non-overlapping windows
    would otherwise give the longest sequence length ~20x fewer training
    examples than the shortest, confounding "sequence length" with "amount
    of training data" instead of isolating the first one, which is what the
    assignment is actually asking about.
    """
    usable = max(n_tokens - seq_len - 1, 1)
    return max(1, usable // max(target_n, 1))


class WindowDataset(Dataset):
    """
    Windows of `seq_len` tokens, each paired with the single word that
    comes right after it. Every window in a dataset built with the same
    seq_len is the same length, so batches need no padding here, unlike
    Task 1 and Task 2.

    `stride` controls how far apart consecutive windows start. stride ==
    seq_len gives non-overlapping windows (fewer of them as seq_len grows);
    a smaller stride gives overlapping windows and more of them. See
    `stride_for_target_count` for why this run uses the latter.
    """

    def __init__(self, tokens, stoi, seq_len, stride=None):
        self.ids = [stoi.get(t, stoi[UNK]) for t in tokens]
        self.words = tokens
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.n = max(0, (len(self.ids) - seq_len - 1) // self.stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        x = torch.tensor(self.ids[start:end], dtype=torch.long)
        y = torch.tensor(self.ids[end], dtype=torch.long)
        context_words = self.words[start:end]
        target_word = self.words[end]
        return x, y, context_words, target_word


def collate(batch):
    xs, ys, contexts, targets = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(contexts), list(targets)
