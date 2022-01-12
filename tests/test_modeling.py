import torch

from cartesius.modeling import ScoreHead


def test_score_head():
    m = ScoreHead(d_in=32, dropout=0)
    x = torch.rand((8, 32), requires_grad=True)
    labels = torch.rand((8,))

    y = m(x)

    (y - labels).sum().backward()  # Backward pass, to check gradient flow
    assert y.size() == (8,)
    assert x.grad is not None
