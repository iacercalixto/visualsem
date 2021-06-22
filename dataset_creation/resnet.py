import torch
import torch.nn as nn
from torchvision.models import resnet152

class ResNet152(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet152(pretrained=True)

        # Freeze the network
        for parameter in resnet.parameters():
            parameter.requires_grad = False

        n_inputs = resnet.fc.in_features

        # Binary classification layer
        resnet.fc = nn.Linear(n_inputs, 2)
        self.resnet = resnet

    def forward(self, batch):
        return self.resnet(batch)
