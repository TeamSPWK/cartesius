from torch import nn


class Transformer(nn.Module):
    def __init__(self, d_model, max_seq_len, n_heads, d_ff, dropout, activation, n_layers):
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


    def forward(self, polygon, mask):
        batch_size, seq_len, _ = polygon.size()
        device = polygon.device

        # Embed polygon's coordinates
        coord_emb = self.coord_embeds(polygon)
        pos_emb = self.position_embeds(torch.arange(seq_len, device=device).repeat((batch_size, 1)))
        emb = coord_emb + pos_emb

        # Encode polygon
        hidden = self.encoder(emb, src_key_padding_mask=~mask)

        # Extract a representation for the whole polygon : just take the first token representation
        poly_feat = hidden[:, 0, :]
        return poly_feat
