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
        d_model (int): Dimension of the final features.
        d_ff (int)): Hidden size of the graph model.
        dropout (float): Dropout for the graph model.
        n_layers (int): Number of layers in the graph model.
    """

    def __init__(self, d_model, d_ff, dropout, n_layers):
        super().__init__()

        self.geom_model = GraphSAGE(in_channels=2,
                                    hidden_channels=d_ff,
                                    num_layers=n_layers,
                                    out_channels=d_model,
                                    dropout=dropout)

    def forward(self, graph):
        hidden = self.geom_model(graph.x, graph.edge_index)
        return global_mean_pool(hidden, graph.batch)
