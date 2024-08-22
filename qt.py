import PyQt5
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QProgressBar, QLineEdit
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
# 实例化模型
model = MLP()
class Earthquake(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('Earthquake')
        self.setGeometry(100, 100, 400, 300)

        self.label = QLabel('输入经纬度和地震种类:',self)
        self.label.move(20, 40)

        self.text_1 = QLineEdit(self)
        self.text_1.move(20, 60)

        self.text_2  = QLineEdit(self)
        self.text_2.move(20, 80)

        self.text_3 = QLineEdit(self)
        self.text_3.move(20, 100)

        self.show_text = QLabel('预测结果',self)
        self.show_text.move(20, 140)

        self.result = QLineEdit(self)
        self.result.move(20, 180)
        #
        self.button = QPushButton('预测', self)
        self.button.move(20, 200)
        self.button.clicked.connect(self.predict)
        self.show()
    def predict(self):
        latitude = float(self.text_1.text())
        longitude = float(self.text_2.text())
        mag_type = float(self.text_3.text())
        result = 4 + predict_earthquake_magnitude(latitude, longitude, mag_type)
        print(result)
        self.result.setText(str(result))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Earthquake()
    sys.exit(app.exec_())