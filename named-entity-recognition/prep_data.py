# -*- coding: utf-8 -*-
"""
Get the CoNLL-2003 data ready for training.

Same dataset as the NER project on my GitHub (Elevvo_NLP_Internship_Task4),
which got it from Kaggle (alaakhaled/conll003-englishversion). Kaggle isn't
set up on this machine, so I pull the same data from Hugging Face instead:
eriktks/conll2003, using its auto-converted parquet branch so it loads
without needing a token or the (now blocked) old-style dataset script.
Same tokens, same tags, same standard train/val/test split sizes
(14,041 / 3,250 / 3,453), just a different pipe to get it.

I keep the original casing. Capitalization is one of the strongest signals
for named entities in English (a capitalized word mid-sentence is much more
likely to be part of a name), so lowercasing here would throw away real
information, the same reason the earlier NER project's README calls out
keeping case and punctuation on purpose.

Usage (from repo root):
    python prep_data.py
"""
from __future__ import annotations

import json
import os

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# Standard CoNLL-2003 tag order, confirmed against the loaded dataset itself
# (checked that ner_tags int 3 -> 'EU' and int 7 -> 'German'/'British' match
# the well-known first sentence of the corpus).
TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def main():
    ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tags.txt"), "w", encoding="utf-8") as f:
        for tag in TAGS:
            f.write(tag + "\n")

    split_map = {"train": "train", "val": "validation", "test": "test"}
    for out_name, hf_split in split_map.items():
        rows = ds[hf_split]
        path = os.path.join(OUT_DIR, f"{out_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                tokens = row["tokens"]
                tags = [TAGS[i] for i in row["ner_tags"]]
                f.write(json.dumps({"tokens": tokens, "tags": tags}, ensure_ascii=False) + "\n")
        n_entities = sum(1 for row in rows for t in row["ner_tags"] if TAGS[t].startswith("B-"))
        print(f"{out_name}: {len(rows)} sentences, {n_entities} entities -> {path}")


if __name__ == "__main__":
    main()
