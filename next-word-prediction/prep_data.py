# -*- coding: utf-8 -*-
"""
Get WikiText-2 ready for next-word prediction.

WikiText-2 is the standard dataset for exactly this comparison: it's what
the original paper on this kind of RNN/GRU/LSTM language modeling benchmark
(Merity et al., the paper that introduced WikiText) uses. Pulled from
Hugging Face (Salesforce/wikitext, config wikitext-2-v1), no login needed.
This is the pre-tokenized version: punctuation is already split into its
own tokens and rare words are already capped to a fixed vocabulary, which
is the standard, literature-comparable setup for word-level language
modeling, so there's no need to write a custom tokenizer here.

Each split becomes one long stream of tokens (articles run together, the
same way the original benchmark treats them). Sequence windows get built
later in dataset.py, once per sequence length, straight from this stream.

Usage (from repo root):
    python prep_data.py
"""
from __future__ import annotations

import os

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    os.makedirs(OUT_DIR, exist_ok=True)

    split_map = {"train": "train", "val": "validation", "test": "test"}
    for out_name, hf_split in split_map.items():
        text = " ".join(row["text"] for row in ds[hf_split])
        tokens = text.split()
        path = os.path.join(OUT_DIR, f"{out_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))
        print(f"{out_name}: {len(tokens):,} tokens -> {path}")


if __name__ == "__main__":
    main()
