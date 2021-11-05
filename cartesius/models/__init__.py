from .transformer import Transformer


def create_model(model_name, conf):
    if model_name == "transformer":
        return Transformer(
            d_model=conf["d_model"],
            max_seq_len=conf["max_seq_len"],
            n_heads=conf["n_heads"],
            d_ff=conf["d_ff"],
            dropout=conf["dropout"],
            activation=conf["activation"],
            n_layers=conf["n_layers"],
        )
    else:
        raise ValueError(f"Unknown model ({model_name}). Please provide an existing model.")
