from .geometric import Geometric
from .se3 import SE3
from .transformer import Transformer


def create_model(model_name, conf):
    if model_name == "transformer":
        return Transformer(
            d_feature=conf["d_feature"],
            d_model=conf["d_model"],
            max_seq_len=conf["max_seq_len"],
            n_heads=conf["n_heads"],
            d_ff=conf["d_ff"],
            dropout=conf["dropout"],
            activation=conf["activation"],
            n_layers=conf["n_layers"],
            pooling=conf["pooling"],
        )
    elif model_name == "graph":
        return Geometric(
            d_feature=conf["d_feature"],
            d_model=conf["d_model"],
            d_ff=conf["d_ff"],
            dropout=conf["dropout"],
            n_layers=conf["n_layers"],
        )
    if model_name == "se3":
        return SE3(
            d_model=conf["d_model"],
            max_seq_len=conf["max_seq_len"],
            n_heads=conf["n_heads"],
            n_layers=conf["n_layers"],
            adjacent_only=conf["adjacent_only"],
        )
    else:
        raise ValueError(f"Unknown model ({model_name}). Please provide an existing model.")
