"""
model.py

Description:
    This module is responsible for creating the model arhcticture

Responsibilities:
    - setting up the model arhcticture

Main Components:
    - Model class
    - forward

Dependencies:
    - torch



"""

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, vocab_size, emb=50, hidden=100, layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb)

        self.rnn = nn.LSTM(
            emb,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden, vocab_size)

        # weight tying
        if emb == hidden:
            self.fc.weight = self.embedding.weight

    def forward(self, x):

        x = self.embedding(x)

        x = self.dropout(x)

        x, _ = self.rnn(x)

        x = self.dropout(x)

        x = self.fc(x)

        return x