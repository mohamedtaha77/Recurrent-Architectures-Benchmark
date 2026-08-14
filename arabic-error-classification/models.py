# -*- coding: utf-8 -*-
"""
One classifier class, parametrized by recurrent cell type.

Fair-comparison requirement from the assignment: everything except the
recurrent architecture must be identical across RNN/GRU/LSTM (same
embedding dim, hidden size, dropout, number of layers, ...). Using a single
class with a `cell` switch, rather than three separate classes, makes that
guarantee structural instead of something to keep in sync by hand.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, cell="lstm", emb_dim=128,
                 hidden_dim=128, num_layers=1, dropout=0.3, pad_idx=0,
                 bidirectional=True):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell!r}, expected one of {list(CELLS)}")
        self.cell_name = cell
        self.bidirectional = bidirectional
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
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, lengths):
        x = self.emb_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        h_n = hidden[0] if self.cell_name == "lstm" else hidden  # (num_layers*dirs, B, H)
        if self.bidirectional:
            # last layer's forward and backward final states, concatenated
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]
        return self.classifier(self.out_dropout(last))

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
