"""
      latitude  longitude  mag  magType
3853   32.7286    76.3907  4.2        1
3886   20.1368   121.3312  4.9        1
2444   24.9153   123.2355  4.7        1
3889   44.3410    79.2948  4.3        1
3616   24.1550    94.3136  4.5        1
Test Loss:  0.13148075342178345
"""

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



def predict_earthquake_magnitude(latitude, longitude, mag_type):
    # 将输入数据转化为张量并确保数据类型为float
    input_data = torch.tensor([[latitude, longitude, mag_type]], dtype=torch.float)

    # 将模型设置为评估模式
    model.eval()

    # 使用模型进行预测
    with torch.no_grad():  # 不需要计算梯度，提高性能
        # 将输入张量传递给模型进行预测
        prediction = model(input_data)

    # 从预测结果张量中获取数值
    predicted_magnitude = prediction.item()

    return predicted_magnitude


# 假设的纬度、经度和震级类型
latitude = 41
longitude = 83
mag_type = 1  # 假设的震级类型，具体数值需要根据数据集实际情况确定

# 调用预测函数
predicted_magnitude = predict_earthquake_magnitude(latitude, longitude, mag_type)

# 打印预测结果
print(f"预测的震级为:{4+predicted_magnitude:.2f}")