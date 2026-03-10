"""
preproccess.py

Description:
    This module is responsible for cleaning and spliting datasets. It also provide vocab ids 
    since we are using text

Responsibilities:
    - convert text into ids for embeddings
    - creating batches to max the training performance

Main Components:
    - DatasetPreprocessing class
    - handle_data

Dependencies:
    - torch
    - src/data_loader.py 


"""
import torch as t 
from torch.utils.data import DataLoader, TensorDataset
from data_loader import DatasetLoader


class DatasetPreprocessing(DatasetLoader):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.df = self.loading_data()
        
    def handle_data(self):
        tokenized_data = []
        for sentence in self.df["data"]:
            tokenized_sent= sentence.lower().split()
            tokenized_data.append(tokenized_sent)
            
        self.df["tokenized_data"] = tokenized_data
        
        # create vocab
        flatten_words = [ word for sentence in tokenized_data for word in sentence]
        vocab = {word:idx for idx,word in enumerate(set(flatten_words))}
        
        seq_length = 17 # this number indicates the number of inputs given to the model 
        
        x = []
        y = []
        
        for sentence in tokenized_data:
            for i in range(len(sentence) - seq_length):
                seq = sentence[i:i+seq_length]     # moving window
                target = sentence[i + seq_length]  # next word after the window
                
                x.append([vocab[word] for word in seq])
                y.append(vocab[target])

        # batches
        x = t.tensor(x)
        y = t.tensor(y)
        
        data = TensorDataset(x , y)
        train_data = DataLoader(data , batch_size=32 , shuffle=False , drop_last=False)
        # that returns a tuple btw!
        
        return train_data , len(vocab)+1, vocab
    




