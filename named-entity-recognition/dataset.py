# -*- coding: utf-8 -*-
"""Shared vocab, dataset, and entity-scoring code for all three NER runs."""
from __future__ import annotations

import json
import os
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def read_jsonl(split):
    rows = []
    with open(os.path.join(DATA_DIR, f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_tags():
    with open(os.path.join(DATA_DIR, "tags.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def build_vocab(rows, min_freq=1):
    counts = Counter()
    for row in rows:
        counts.update(row["tokens"])
    itos = [PAD, UNK] + [w for w, c in counts.items() if c >= min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


class NERDataset(Dataset):
    def __init__(self, rows, stoi, tag2idx):
        self.rows = rows
        self.stoi = stoi
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in row["tokens"]]
        tag_ids = [self.tag2idx[t] for t in row["tags"]]
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long),
                row["tokens"], row["tags"])


def collate(batch, pad_tag_idx):
    seqs, tags, tokens, gold_tags = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    maxlen = int(lengths.max())
    padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    padded_tags = torch.full((len(seqs), maxlen), pad_tag_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(seqs, tags)):
        padded[i, :len(s)] = s
        padded_tags[i, :len(t)] = t
    return padded, padded_tags, lengths, tokens, gold_tags


# ---------------------------------------------------------------------------
# Entity-level scoring (a small stand-in for seqeval, which would not
# install in this environment because of a broken build dependency).
# Standard exact-match entity F1: an entity is (start, end, type), and a
# prediction only counts if the span and the type both match the gold
# entity exactly.
# ---------------------------------------------------------------------------
def bio_to_spans(tags):
    spans = []
    start, label = None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start, label = i, tag[2:]
        elif tag.startswith("I-") and label == tag[2:]:
            pass
        else:
            if start is not None:
                spans.append((start, i, label))
            start, label = None, None
            if tag.startswith("I-"):
                start, label = i, tag[2:]
    return spans


def entity_scores(gold_tag_seqs, pred_tag_seqs):
    tp = fp = fn = 0
    by_type = {}
    for gold_tags, pred_tags in zip(gold_tag_seqs, pred_tag_seqs):
        gold_spans = set(bio_to_spans(gold_tags))
        pred_spans = set(bio_to_spans(pred_tags))
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
        for span in gold_spans | pred_spans:
            label = span[2]
            d = by_type.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            if span in gold_spans and span in pred_spans:
                d["tp"] += 1
            elif span in pred_spans:
                d["fp"] += 1
            else:
                d["fn"] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    overall = prf(tp, fp, fn)
    per_type = {label: prf(**{k: v for k, v in d.items()}) for label, d in sorted(by_type.items())}
    return overall, per_type
