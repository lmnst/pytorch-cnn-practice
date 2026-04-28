import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
from model import LeNet

def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                              download=True,)
    
    # test_data, val_data = Data.random_split(train_data, lengths=[round(0.8*len(train_data)), round(0.2*len(train_data))])

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0,)
    
    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # init para
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # x -- tensor; y -- label
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 训练模式
            model.eval()

            # 前向传播， output是十个神经元的值，需要softmax得出最大概率
            output = model(test_data_x)

            # find the biggest value and return the label
            pre_label = torch.argmax(output, dim=1)

            # 
            test_corrects += torch.sum(pre_label == test_data_y.data)
            test_num += test_data_x.size(0)

    # acc
    test_acc = test_corrects.double().item()/test_num

    print("test_acc:", test_acc)


if __name__ == "__main__":
    model = LeNet()

    model.load_state_dict(torch.load("best model path"))

    test_dataload = test_data_process()

    # test_model_process(model, test_dataload)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    with torch.no_grad():
        for b_x, b_y in test_dataload:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)

            pre_label = torch.argmax(output, dim=1)
            # 张量形式（tensor， pytorch，）.item(),取数值
            result = pre_label.item()
            label = b_y.item()

            print("预测值：", result, "------", "标签值：", label)

