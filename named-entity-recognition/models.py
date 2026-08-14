# -*- coding: utf-8 -*-
"""
One tagger class, parametrized by recurrent cell type, same idea as Task 1's
SequenceClassifier. The difference from Task 1: NER needs a label for every
token, not one label for the whole sentence, so the classifier head runs on
the recurrent layer's full output sequence instead of just its last hidden
state.
"""
from __future__ import annotations

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class TokenTagger(nn.Module):
    def __init__(self, vocab_size, num_tags, cell="lstm", emb_dim=128,
                 hidden_dim=128, num_layers=1, dropout=0.3, pad_idx=0,
                 bidirectional=True):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell!r}, expected one of {list(CELLS)}")
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = CELLS[cell](
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_tags)

    def forward(self, input_ids, lengths):
        x = self.emb_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
        return self.classifier(self.out_dropout(out))

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
