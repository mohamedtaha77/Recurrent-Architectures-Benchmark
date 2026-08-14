# -*- coding: utf-8 -*-
"""
Build the text-classification dataset from real QALB data.

Uses this project's own edit-extraction/classification pipeline
(qalb_diff.py: extract_edits + classify) instead of hand-labeling or
inventing a new heuristic. Each QALB source/target sentence pair is diffed
into edits; each edit is typed (hamza_alef, alef_maqsura, punctuation, ...)
by the same classify() already cross-checked against real QALB character
confusions. A short token window around each edit becomes one
classification example, labeled by that edit's type.

Why windows, not whole sentences: QALB "sentences" are paragraph-scale
(~12.5 edits/sentence on average in 2014-L1-Train), so requiring exactly one
edit per whole sentence would throw away almost all the data. A window keeps
each example short and still gives the model real local context.

Split at the sentence level (not the window level) so windows from the same
source sentence never leak across train/val/test.

Needs a local copy of the QALB release under data/raw/qalb/ (see the
README: QALB requires free registration and its license forbids
redistribution, so it isn't bundled in this repo).

Usage (from this folder):
    python prep_data.py
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict

from qalb_diff import tokenize, extract_edits, classify

QALB_ROOT = "data/raw/qalb/extracted/QALB-0.9.1-Dec03-2021-SharedTasks/data"
SPLITS = [
    (f"{QALB_ROOT}/2014/train/QALB-2014-L1-Train.m2",
     f"{QALB_ROOT}/2014/train/QALB-2014-L1-Train.cor"),
    (f"{QALB_ROOT}/2015/train/QALB-2015-L2-Train.m2",
     f"{QALB_ROOT}/2015/train/QALB-2015-L2-Train.cor"),
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

WINDOW = 6          # tokens of context on each side of the edit span
TOP_K = 6            # number of error-type classes to keep
MIN_CLASS_COUNT = 300
MAX_PER_CLASS = {"train": 4000, "val": 500, "test": 500}

# "spelling/morph" is this project's own documented catch-all bucket for
# edits that don't fit a cleaner pattern (see src/evaluate.py's classify()
# and CLAUDE.md's discussion of it) - structurally noisier than the other
# classes, and it measured far worse than every other class in both the
# word-level and char-level runs (F1 ~0.27-0.47 vs 0.5-0.96 elsewhere).
# Swapped for ta_marbuta (ة/ه), a clean single-character-swap class with
# plenty of real examples (9,141 in the full L1+L2 pool).
EXCLUDE_LABELS = {"spelling/morph"}

# Sentinels bracketing the edited span inside each window, so the model
# doesn't have to guess which of the ~13 tokens is the erroneous one.
# Chosen outside the Arabic block so they can never collide with real text.
EDIT_START, EDIT_END = "✁", "✂"


def load_m2_sources(m2_path):
    out = []
    with open(m2_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("S "):
                out.append(line[2:].strip())
    return out


def load_cor_targets(cor_path):
    out = []
    with open(cor_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                line = line[2:]
            out.append(line.strip())
    return out


def load_pairs():
    pairs = []
    for m2_path, cor_path in SPLITS:
        src = load_m2_sources(m2_path)
        tgt = load_cor_targets(cor_path)
        assert len(src) == len(tgt), f"{m2_path}: {len(src)} src vs {len(tgt)} tgt"
        for s, t in zip(src, tgt):
            if s and t:
                pairs.append((s, t))
    return pairs


def window_examples(pairs, seed=42):
    """For every sentence, extract (window_text, label) for every edit."""
    rng = random.Random(seed)
    by_sentence = []  # list of (split, [(window_text, label), ...])
    for s, t in pairs:
        split = "train" if rng.random() < 0.8 else ("val" if rng.random() < 0.5 else "test")
        st, tt = tokenize(s), tokenize(t)
        edits = extract_edits(st, tt)
        examples = []
        for e in edits:
            label = classify(st, e)
            lo = max(0, e.i - WINDOW)
            hi = min(len(st), e.j + WINDOW)
            marked = st[lo:e.i] + [EDIT_START] + st[e.i:e.j] + [EDIT_END] + st[e.j:hi]
            examples.append((" ".join(marked), label))
        if examples:
            by_sentence.append((split, examples))
    return by_sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print("loading real QALB pairs...", flush=True)
    pairs = load_pairs()
    print(f"  {len(pairs)} source/target pairs", flush=True)

    print("extracting edits + windows...", flush=True)
    by_sentence = window_examples(pairs, seed=a.seed)

    label_counts = Counter()
    for _, examples in by_sentence:
        for _, label in examples:
            label_counts[label] += 1
    print("full label distribution:")
    for label, n in label_counts.most_common():
        print(f"  {label:20s} {n}")

    eligible = [(l, n) for l, n in label_counts.most_common() if n >= MIN_CLASS_COUNT and l not in EXCLUDE_LABELS]
    top_labels = [l for l, n in eligible[:TOP_K]]
    print(f"\nkeeping top {len(top_labels)} classes: {top_labels}")

    pooled = defaultdict(lambda: defaultdict(list))  # split -> label -> [text,...]
    for split, examples in by_sentence:
        for text, label in examples:
            if label in top_labels:
                pooled[split][label].append(text)

    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    counts_report = {}
    for split in ("train", "val", "test"):
        rows = []
        counts_report[split] = {}
        for label in top_labels:
            texts = pooled[split][label]
            rng.shuffle(texts)
            cap = MAX_PER_CLASS[split]
            texts = texts[:cap]
            counts_report[split][label] = len(texts)
            for text in texts:
                rows.append((text, label))
        rng.shuffle(rows)
        with open(os.path.join(OUT_DIR, f"{split}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            for text, label in rows:
                text = text.replace("\t", " ")
                f.write(f"{text}\t{label}\n")
        print(f"{split}: {len(rows)} examples -> {counts_report[split]}")

    with open(os.path.join(OUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for label in top_labels:
            f.write(label + "\n")

    print(f"\nwrote dataset -> {OUT_DIR}/{{train,val,test}}.tsv, labels.txt")


if __name__ == "__main__":
    main()
