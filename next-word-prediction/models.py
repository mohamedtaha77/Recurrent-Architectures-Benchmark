# -*- coding: utf-8 -*-
"""
One next-word model class, parametrized by recurrent cell type, same idea
as Task 1 and Task 2.

This one has to run one direction only. Task 1 and Task 2 could read the
whole input before deciding anything, so a bidirectional layer was fair
game there. Predicting the next word can only use words that came before
it. Reading the answer's own future words in would defeat the task.
"""
from __future__ import annotations

import torch.nn as nn

CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class WordLM(nn.Module):
    def __init__(self, vocab_size, cell="lstm", emb_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.3, pad_idx=0):
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
        )
        self.out_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.emb_dropout(self.embedding(input_ids))
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.classifier(self.out_dropout(last))

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
