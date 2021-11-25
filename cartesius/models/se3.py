import torch
from torch import nn

try:
    # Optional, have to install additional dependencies for this to work
    from se3_transformer_pytorch import SE3Transformer
except ImportError:
    pass


class SE3(nn.Module):
    """Model using SE(3)-Transformer (invariant to translation and rotations).

    Args:
        d_model (int): Dimension for the Transformer.
        max_seq_len (int): Maximum sequence length.
        n_heads (int): Number of attention heads for the Transformer.
        n_layers (int): Number of layers in the Transformer Encoder.
    """

    def __init__(self, d_model, max_seq_len, n_heads, n_layers):
        super().__init__()

        # SE(3)-Transformer
        self.encoder = SE3Transformer(
            dim=d_model,
            depth=n_layers,
            heads=n_heads,
            num_degrees=1,
            num_positions=max_seq_len,
            num_tokens=1,
            differentiable_coors=True,
        )

    def forward(self, polygon, mask):
        batch_size, seq_len, _ = polygon.size()

        # Ensure the polygon's coordinates are 3D
        z = polygon.new_zeros((batch_size, seq_len, 1))
        polygon = torch.cat([polygon, z], dim=-1)

        # Create features (all points are the same feature)
        feat = torch.zeros((batch_size, seq_len), dtype=torch.int)

        return self.encoder(feat, polygon, mask, return_pooled=True)
