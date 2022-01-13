import torch
from torch import nn


class Transformer(nn.Module):
    """Basic Transformer implementation for Cartesius benchmark.

    Args:
        d_model (int): Dimension for the Transformer Encoder Layer.
        max_seq_len (int): Maximum sequence length.
        n_heads (int): Number of attention heads for the Transformer Encoder Layer.
        d_ff (int): Hidden size of the FF network in the Transformer Encoder Layer.
        dropout (float): Dropout for the Transformer Encoder Layer.
        activation (str): Activation function to use in the Transformer Encoder Layer.
        n_layers (int): Number of layers in the Transformer Encoder.
        pooling (str): Type of pooling to apply to extract the polygon representation
            (`first` for using the first token as polygon representation, `mean`
            for mean pooling over all tokens).
    """

    def __init__(self, d_model, max_seq_len, n_heads, d_ff, dropout, activation, n_layers, pooling):
        super().__init__()

        # Embeddings
        self.coord_embeds = nn.Linear(2, d_model, bias=False)
        self.position_embeds = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=n_heads,
                                                    dim_feedforward=d_ff,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # Pooling operation for polygon representation
        assert pooling in ["first", "mean", "max"]
        self.pooling = pooling

    def forward(self, polygon, mask):
        batch_size, seq_len, _ = polygon.size()
        device = polygon.device

        # Embed polygon's coordinates
        coord_emb = self.coord_embeds(polygon)
        pos_emb = self.position_embeds(torch.arange(seq_len, device=device).repeat((batch_size, 1)))
        emb = coord_emb + pos_emb

        # Encode polygon
        hidden = self.encoder(emb, src_key_padding_mask=~mask)

        # Extract a representation for the whole polygon
        if self.pooling == "first":
            poly_feat = hidden[:, 0, :]
        elif self.pooling == "mean":
            poly_feat = torch.mean(hidden, dim=1)
        elif self.pooling == "max":
            poly_feat = torch.max(hidden, dim=1)[0]
        return poly_feat
