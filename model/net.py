
from torch import nn as nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1)

        self.batchnorm1 = nn.BatchNorm2d(8)

        self.relu = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, padding=2, stride=1)

        self.batchnorm2 = nn.BatchNorm2d(32)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=1568, out_features=600)

        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=600, out_features=10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = out.view(-1, 1568)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
