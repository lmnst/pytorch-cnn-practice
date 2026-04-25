import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
from model import LeNet

def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor]),
                              download=True,)
    
    train_data, val_data = Data.random_split(train_data, lengths=[round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8,)
    
    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8,)
    
    return train_dataloader, val_dataloader