"""
model.py

Description:
    This module is responsible for predicting 

Responsibilities:
    - predicting the next token 

Main Components:


Dependencies:
    - torch



"""
from train import Train
from preproccess import DatasetPreprocessing
import torch as t


obj = DatasetPreprocessing("data.json")
trn = Train()  # already has trained model

class Prediction:
    def __init__(self, prompt, vocab, seq_length=4, steps=2):
        self.model = trn.model
        self.vocab = vocab
        self.inv_vocab = {v:k for k,v in vocab.items()}
        self.seq_length = seq_length
        self.prompt = [w.lower() for w in prompt]  # make lowercase
        self.steps = steps

    def generate(self):
        input_ids = [self.vocab[w] if w in self.vocab else 0 for w in self.prompt]
        x = t.tensor([input_ids[-self.seq_length:]])  # ensure <= seq_length
        generated = self.prompt.copy()
        
        self.model.eval()
        with t.no_grad():
            print(self.model)
            for _ in range(self.steps):
                output = self.model(x)
                last_output = output[:, -1, :]
                probs = t.softmax(last_output, dim=-1)
                temperature = 0.7
                probs = t.softmax(last_output / temperature, dim=-1)
                predicted_id = t.multinomial(probs, 1).item()
                generated.append(self.inv_vocab.get(predicted_id, "<UNK>"))
                x = t.tensor([[predicted_id]])
        
        return " ".join(generated)

# Example usage


train_data , vocab_size , vocab = obj.handle_data()
pred = Prediction(["Hello" , "how" ], vocab)
print(pred.generate())