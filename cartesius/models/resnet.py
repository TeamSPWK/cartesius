import torch
from torch import nn


class ResNet(nn.Module):
    """Model using SE(3)-Transformer (invariant to translations and rotations).

    Args:
        d_model (int): Dimension for the Transformer.
        max_seq_len (int): Maximum sequence length.
        n_heads (int): Number of attention heads for the Transformer.
        n_layers (int): Number of layers in the Transformer Encoder.
        adjacent_only (bool): If set to `True`, use Adjacency matrix in Transformer. If
            set to False, all nodes are attended.
    """

    def __init__(self, d_model):
        super().__init__()

        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.tail = nn.Linear(1000, d_model)

    def forward(self, x):
        out1 = self.resnet18(x)
        out2 = self.tail(out1)
        return out2
