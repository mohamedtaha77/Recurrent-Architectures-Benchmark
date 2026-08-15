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

import argparse
import os

from datasets import load_dataset

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="wikitext-2-v1",
                     help="Hugging Face Salesforce/wikitext config, e.g. wikitext-2-v1 "
                          "or wikitext-103-v1 for the ~50x larger sibling corpus.")
    ap.add_argument("--out-dir", default=None,
                     help="Defaults to ./data; pass a different path (e.g. data/wikitext103) "
                          "to avoid overwriting the data an existing run was trained on.")
    a = ap.parse_args()
    out_dir = a.out_dir or DEFAULT_OUT_DIR

    ds = load_dataset("Salesforce/wikitext", a.config)
    os.makedirs(out_dir, exist_ok=True)

    split_map = {"train": "train", "val": "validation", "test": "test"}
    for out_name, hf_split in split_map.items():
        # Stream row by row instead of building one giant string + token list
        # in memory: fine at WikiText-2's ~2M tokens, but WikiText-103's
        # ~103M tokens as individual Python string objects is several GB and
        # was blowing out memory.
        path = os.path.join(out_dir, f"{out_name}.txt")
        n_tokens = 0
        with open(path, "w", encoding="utf-8") as f:
            wrote_any = False
            for row in ds[hf_split]:
                tokens = row["text"].split()
                if not tokens:
                    continue
                if wrote_any:
                    f.write(" ")
                f.write(" ".join(tokens))
                wrote_any = True
                n_tokens += len(tokens)
        print(f"{out_name}: {n_tokens:,} tokens -> {path}")


if __name__ == "__main__":
    main()
