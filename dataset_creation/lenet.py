#https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self, width=300): # Initialize lenet according to architecture + image width
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(int(16*((((width - 4)/2) - 4)/2)**2), 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 2)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.drop1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.drop2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
