import torch
from torch import nn

try:
    # Optional, have to install additional dependencies for this to work
    from se3_transformer_pytorch import SE3Transformer
except ImportError:
    pass


class SE3(nn.Module):
    """Model using SE(3)-Transformer (invariant to translations and rotations).

    Args:
        d_model (int): Dimension for the Transformer.
        max_seq_len (int): Maximum sequence length.
        n_heads (int): Number of attention heads for the Transformer.
        n_layers (int): Number of layers in the Transformer Encoder.
        adjacent_only (bool): If set to `True`, use Adjacency matrix in Transformer. If
            set to False, all nodes are attended.
    """

    def __init__(self, d_model, max_seq_len, n_heads, n_layers, adjacent_only):
        super().__init__()

        self.adj = adjacent_only

        # SE(3)-Transformer
        self.encoder = SE3Transformer(
            dim=d_model,
            depth=n_layers,
            heads=n_heads,
            num_degrees=1,
            num_positions=max_seq_len,
            num_tokens=1,
            differentiable_coors=True,
            attend_sparse_neighbors=adjacent_only,
            num_neighbors=0 if adjacent_only else float("inf"),
        )

    def forward(self, polygon, mask):
        batch_size, seq_len, _ = polygon.size()
        device = polygon.device

        # Ensure the polygon's coordinates are 3D
        z = polygon.new_zeros((batch_size, seq_len, 1))
        polygon = torch.cat([polygon, z], dim=-1)

        # Create features (all points are the same feature)
        feat = torch.zeros((batch_size, seq_len), dtype=torch.int, device=device)

        if self.adj:
            # Create adjency matrix (a polygon is just a long chain of points)
            i = torch.arange(seq_len, device=device)
            adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))
        else:
            adj_mat = None

        return self.encoder(feat, polygon, mask, adj_mat=adj_mat, return_pooled=True)
