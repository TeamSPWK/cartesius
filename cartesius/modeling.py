import torch
from torch import nn


class ScoreHead(nn.Module):
    """Fully connected Head to compute a score.

    Args:
        d_in (int): Input dimension.
        d_hid (int): Hidden dimension.
        dropout (float): Dropout rate.
        n (int, optional): output dimension. Default to 1.
    """

    def __init__(self, d_in, d_hid, dropout, n=1):
        super().__init__()
        self.dense = nn.Linear(d_in, d_hid)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(d_hid, n)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.squeeze(-1)
