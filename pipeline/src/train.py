"""
train.py

Description:
    This module is responsible for training

Responsibilities:
    - getting data from preproccess

Main Components:
    - Train class
    - train function

Dependencies:
    - torch
    - src/preproccess.py


"""

from preproccess import DatasetPreprocessing
from model import Model
import torch as t
from torch import nn

obj = DatasetPreprocessing("data.json")

train_data , vocab_size , vocab = obj.handle_data()

model = Model(vocab_size)
optim = t.optim.Adam(model.parameters() , lr=0.01)
c = nn.CrossEntropyLoss()
print(vocab_size)

class Train:
    def __init__(self):
        super().__init__()
        self.model = model
        self.optim = optim
        self.loss = c
        self.train_data = train_data
    def train(self):
        epochs = 100  # more epochs for overfitting

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in self.train_data:
                self.optim.zero_grad()
                
                output = self.model(batch_x)       # [batch, seq_len, vocab_size]
                last_output = output[:, -1, :]     # take last timestep
                
                loss = self.loss(last_output, batch_y.long())  # target = [batch]
                
                loss.backward()
                self.optim.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(self.train_data):.4f}")

        return self.model
