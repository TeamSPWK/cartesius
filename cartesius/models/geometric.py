from torch import nn

try:
    # Optional, have to install additional dependencies for this to work
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.nn import GraphSAGE
except ImportError:
    pass


class Geometric(nn.Module):
    """Basic graph-based model, using already-implemented models from torch
    Geometric and extracting a graph representation as features.

    Args:
        d_feature (int): Dimensions of the input features.
        d_model (int): Dimension of the final features.
        d_ff (int)): Hidden size of the graph model.
        dropout (float): Dropout for the graph model.
        n_layers (int): Number of layers in the graph model.
    """

    def __init__(self, d_feature, d_model, d_ff, dropout, n_layers):
        super().__init__()

        self.geom_model = GraphSAGE(in_channels=d_feature,
                                    hidden_channels=d_ff,
                                    num_layers=n_layers,
                                    out_channels=d_model,
                                    dropout=dropout)

    def forward(self, x, edge_index, batch_index):
        hidden = self.geom_model(x, edge_index)
        return global_mean_pool(hidden, batch_index)
