import torch
import torch.nn as nn
from torchvision.models import vgg19_bn

class VGG19_BN(nn.Module): # With batch normalization

    def __init__(self):
        super().__init__()
        vgg = vgg19_bn(pretrained=True)

        # Freeze the network
        for parameter in vgg.parameters():
            parameter.requires_grad = False

        # Remove the last layer and replace the classifier
        features = list(vgg.classifier.children())[:-1]
        features.append(nn.Linear(4096, 2))
        vgg.classifier = nn.Sequential(*features)
        self.vgg = vgg

    def forward(self, batch):
        return self.vgg(batch)
