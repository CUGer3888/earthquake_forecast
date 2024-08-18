
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset, DataLoader
import warnings
# with open("query.csv") as f:
#     data = pd.read_csv(f)
#     new_data = shuffle(data)
#     datas = new_data[['latitude','longitude','mag','magType']]
    #对latitude进行一定分类
    # datas.loc[:, 'latitude'] = datas['latitude'].apply(lambda x:
    #                                                              1 if x < 10 else
    #                                                              2 if x < 20 else
    #                                                              3 if x < 30 else
    #                                                              4 if x < 40 else
    #                                                              5 if x < 50 else
    #                                                              6 if x < 60 else
    #                                                              7 if x < 70 else
    #                                                              8 if x < 80 else
    #                                                              9 if x < 90 else
    #                                                              10 if x < 100 else
    #                                                              11 if x < 110 else
    #                                                              12 if x < 120 else
    #                                                              13
    #                                                              )
    # datas.loc[:, 'longitude'] = datas['longitude'].apply(lambda x:
    #                                                                1 if x < 10 else
    #                                                                2 if x < 20 else
    #                                                                3 if x < 30 else
    #                                                                4 if x < 40 else
    #                                                                5 if x < 50 else
    #                                                                6 if x < 60 else
    #                                                                7 if x < 70 else
    #                                                                8 if x < 80 else
    #                                                                9 if x < 90 else
    #                                                                10 if x < 100 else
    #                                                                11 if x < 110 else
    #                                                                12 if x < 120 else
    #                                                                13
    #                                                                )
    # datas.loc[:, 'magType'] = datas['magType'].apply(lambda x:
    #                                                            1 if x == 'mb' else
    #                                                            2 if x == 'mww'else
    #                                                            3 if x == 'mwr' else
    #                                                            4)
    # #保存文件
    # datas.to_csv('new.csv', index=False)
df = pd.read_csv("C:\\Users\\CUGac\\PycharmProjects\\astar\\.venv\\Scripts\\earthquake\\new.csv", encoding='gbk')
# 数据预处理,shuffle处理
new_df = df

new_df = shuffle(new_df)
print(new_df.head())

x = new_df[['latitude','longitude','magType']]
y = new_df['mag']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 数据类型转换
x_train_tensor = torch.from_numpy(x_train.to_numpy()).float()
y_train_tensor = torch.from_numpy(y_train.to_numpy()).float()

x_test_tensor = torch.from_numpy(x_test.to_numpy()).float()
y_test_tensor = torch.from_numpy(y_test.to_numpy()).float()

# 创建数据加载器
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)
# 定义一个简单的MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # 添加Batch Normalization
            nn.Dropout(p=0.5),    # 添加Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# 实例化模型
model = MLP()

# 设置损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels.view(-1, 1))

        # 反向传播和优化
        loss.backward()
        optimizer.step()

# 评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        # 这里没有准确率的概念，但是可以计算预测和真实值的差异
        loss = criterion(outputs, labels.view(-1, 1))
    print('Test Loss: ', loss.item())
# 保存模型
torch.save(model.state_dict(), 'mlp_model.pth')
# def predict_earthquake_magnitude(latitude, longitude, mag_type):
#     # 将输入数据转化为张量并确保数据类型为float
#     input_data = torch.tensor([[latitude, longitude, mag_type]], dtype=torch.float)
#
#     # 将模型设置为评估模式
#     model.eval()
#
#     # 使用模型进行预测
#     with torch.no_grad():  # 不需要计算梯度，提高性能
#         # 将输入张量传递给模型进行预测
#         prediction = model(input_data)
#
#     # 从预测结果张量中获取数值
#     predicted_magnitude = prediction.item()
#
#     return predicted_magnitude
#
#
# # 假设的纬度、经度和震级类型
# latitude = 39.9042
# longitude = 116.4074
# mag_type = 1  # 假设的震级类型，具体数值需要根据数据集实际情况确定
#
# # 调用预测函数
# predicted_magnitude = predict_earthquake_magnitude(latitude, longitude, mag_type)
#
# # 打印预测结果
# print(f"预测的震级为: {predicted_magnitude}")