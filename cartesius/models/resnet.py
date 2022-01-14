import torch
from torch import nn


class ResNet(nn.Module):
    """Model using ResNet (Pytorch official model with a dimension reducing tail)

    Args:
        d_model (int): Out dimension for the ResNet.
    """

    def __init__(self, d_model):
        super().__init__()

        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.tail = nn.Linear(1000, d_model)

    def forward(self, x):
        out1 = self.resnet18(x)
        out2 = self.tail(out1)
        return out2
