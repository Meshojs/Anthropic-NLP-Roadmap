"""
data_loader.py

Description:
    This module is responsible for loading datasets for the training and evaluation
    pipelines. It ensures that data from various sources is standardized and ready
    for preprocessing and modeling.

Responsibilities:
    - Read raw data from CSV, JSON
    - Validate the integrity of the data
    - Convert data into structured format (e.g., Pandas DataFrame )

Main Components:
    - DatasetLoader class
    - loading_data function

Dependencies:
    - pandas
    - numpy

Usage:
    - Used by training pipeline to fetch datasets
    - Can be used by inference pipeline for loading new data for predictions

Notes:
    - Assumes raw data files are UTF-8 encoded
    - Does not perform heavy preprocessing; preprocessing should be handled separately
"""

import pandas as pd
import os

class DatasetLoader :
    def __init__(self, file_name:str):
        self.file_name = file_name
        self.formats = ["csv" , "json"]
        self.df = None
    
    def loading_data(self):
        
        assert len(self.file_name) != 0 
        if not os.path.exists(self.file_name):
            raise FileNotFoundError(f"File not found: {self.file_name}")
        
        
        if self.file_name.split(".")[-1] in self.formats:
            self.df = pd.read_json(self.file_name)
            return self.df
        
        elif self.file_name.split(".")[-1] in self.formats:
            self.df = pd.read_csv(self.file_name)
            return self.df
        else : 
            raise Exception("Unknown provided file extensions. [json or  csv].")
    

