import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhysicsDataset(Dataset):
    
    def __init__(self, data, labels,train_state = 'train'):
        
        self.data=data
        self.labels=labels
        
        if train_state == 'train':
            data_train, data_valid, target_train, target_valid = train_test_split(self.data, self.labels,
                                                                                  test_size=0.2, random_state=42)

 
            self.data = data_train
            self.labels = target_train
            print(len(self.data))
            
        elif train_state == 'valid':
            data_train, data_valid, target_train, target_valid = train_test_split(self.data, self.labels,
                                                                                  test_size=0.2, random_state=42)



         
            self.data = data_valid
            self.labels = target_valid
            print(len(self.data)) 
            
        elif train_state=='test':
            print(len(self.data))
            
            
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data)
    
    
    def getitem(self, index):

        x = torch.tensor(self.data[index],dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x,y
        