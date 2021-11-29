import torch

from cartesius.modeling import ScoreHead
from cartesius.models import Geometric
from cartesius.models import SE3
from cartesius.models import Transformer


def test_score_head():
    m = ScoreHead(d_in=32, d_hid=16, dropout=0)
    x = torch.rand((8, 32), requires_grad=True)
    labels = torch.rand((8,))

    y = m(x)

    (y - labels).sum().backward()  # Backward pass, to check gradient flow
    assert y.size() == (8,)
    assert x.grad is not None


def test_transformer():
    m = Transformer(d_model=32, max_seq_len=64, n_heads=8, d_ff=64, dropout=0, activation="gelu", n_layers=2)
    x = torch.rand((8, 36, 2), requires_grad=True)
    mask = torch.rand((8, 36)) < 0.5
    labels = torch.rand((8,))

    y = m(x, mask)

    (y.sum(-1) - labels).sum().backward()  # Backward pass, to check gradient flow
    assert y.size() == (8, 32)
    assert x.grad is not None


def test_geometric():
    m = Geometric(d_model=32, d_ff=64, dropout=0, n_layers=2)
    x = torch.rand((36, 2), requires_grad=True)
    e = torch.stack([torch.arange(36), (torch.arange(36) + 1) % 36])
    b = torch.tensor([0] * 20 + [1] * 10 + [2] * 5 + [3])
    labels = torch.rand((4,))

    y = m(x, e, b)

    (y.sum(-1) - labels).sum().backward()  # Backward pass, to check gradient flow
    assert y.size() == (4, 32)
    assert x.grad is not None


def test_se3_transformer():
    m = SE3(d_model=4, max_seq_len=64, n_heads=2, n_layers=2, adjacent_only=True)
    x = torch.rand((8, 36, 2), requires_grad=True)
    mask = torch.rand((8, 36)) < 0.5
    labels = torch.rand((8,))

    y = m(x, mask)

    (y.sum(-1) - labels).sum().backward()  # Backward pass, to check gradient flow
    assert y.size() == (8, 4)
    assert x.grad is not None
