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

def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
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

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    #判断设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #损失函数使用交叉熵
    loss_function = nn.CrossEntropyLoss()
    #使用device训练模型
    model = model.to(device)
    #最佳模型参数保存
    best_model_wts = copy.deepcopy(model.state_dict())

    #init parameter
    #最高准确度
    best_acc = 0.0
    #训练集准损失列表
    train_loss_all = []
    #验证集损失列表
    val_loss_all = []
    #训练集准确度列表
    train_acc_all = []
    #验证集准确度列表
    val_acc_all = []

    since = time.time()


    for epoch in range (num_epochs):
        print("EOPCH {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # init
        #训练集损失
        train_loss = 0.0
        #训练集准确度
        train_corrects = 0
        #验证集
        val_loss = 0.0
        val_corrects = 0
        #训练集样本数量
        train_num = 0
        val_num = 0

        # b_x 是当前批次的输入数据，它是一个张量（Tensor），形状通常是 [batch_size, 通道数, 高, 宽]
        for step, (b_x, b_y) in enumerate(train_dataloader):
            #将特征放入设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #模型设置为训练模式
            model.train()
            # 前向传播过程，输入一个batch，输出为一个batch中的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每个batch的损失
            loss = loss_function(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()

            # 反向传播计算
            loss.backward()

            # 利用梯度下降更新weight
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            # 1.train_corrects这个变量是计算每个批次当中正确预测的数量的
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 2.train_num是计算到目前为止一共处理了多少样本
            train_num += b_x.size(0)
            # 3.这俩参数一比应当就能算出截止某次训练时的准确率了

        model.eval()
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_dataloader):
                #将特征放入验证设备
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                # 前向传播得到结果
                output = model(b_x)

                # 查找每一行中最大值对应的行标
                pre_lab = torch.argmax(output, dim=1)

                # loss
                loss = loss_function(output, b_y)
                # 
                val_loss += loss.item() * b_x.size(0)
                # 1.train_corrects这个变量是计算每个批次当中正确预测的数量的
                val_corrects += torch.sum(pre_lab == b_y.data)
                # 2.train_num是计算到目前为止一共处理了多少样本
                val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前的最高准确度
            best_acc = val_acc_all[-1]

            # 保存最佳权重参数
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since

        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    model.load_state_dict(best_model_wts)
    torch.save(model.load_state_dict(best_model_wts), 'LeNet/best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_loss_all":val_loss_all,
                                       "val_acc_all":val_acc_all,})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label = "train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label = "val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label = "train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label = "val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

if __name__ == "__main__":
    # 模型实例化
    my_model = LeNet()

    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(my_model, train_dataloader, val_dataloader, 20)

    matplot_acc_loss(train_process)