from shapely.geometry import Polygon
import torch

try:
    # Optional, have to install additional dependencies for this to work
    from torch_geometric.data import Batch
    from torch_geometric.data import Data
except ImportError:
    pass

PAD_COORD = (0, 0)


class Tokenizer:
    """Base class for Tokenizer.

    Tokenizers takes as input a polygon or a batch of polygons and return a
    dict of tensor representing the polygon(s).

    Sub classes should overwrite the method `tokenize()`.
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass

    def __call__(self, polygons):
        """Main method of the tokenizer. It tokenize the given polygon(s).

        If the input is a list of polygons, polygons will be tokenized as a
        batch.

        Args:
            polygons (list or shapely.geometry.Geometry): Things to tokenize.

        Returns:
            dict: dict of tensors containing the tokenized polygon(s).
        """
        if isinstance(polygons, list):
            # Batched input
            return self.tokenize(polygons)
        else:
            # Simulate a batched input
            return self.tokenize([polygons])

    def tokenize(self, polygons):
        """Method tokenizing a list of polygons into tensors.

        Args:
            polygons (list): List of polygons to tokenize.

        Raises:
            NotImplementedError: Exception raised if the method was not
                over-written in the sub class.

        Returns:
            dict: dict of tensors containing the tokenized polygons.
        """
        raise NotImplementedError


class TransformerTokenizer(Tokenizer):
    """Tokenizer for Transformer model.

    This is a basic tokenizer, used with Transformer model. It just uses the coordinates
    of the polygon and pad them appropriately.

    Args:
        max_seq_len (int): Maximum sequence length. An exception will be raised if you
            try to tokenize a polygon with more points than this.
    """

    def __init__(self, max_seq_len, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()

        self.max_seq_len = max_seq_len

    def tokenize(self, polygons):
        poly_coords = [list(p.boundary.coords) if isinstance(p, Polygon) else list(p.coords) for p in polygons]
        pad_size = max(len(p_coords) for p_coords in poly_coords)

        if pad_size > self.max_seq_len:
            raise RuntimeError(f"Polygons are too big to be tokenized ({pad_size} > {self.max_seq_len})")

        masks = []
        tokens = []
        for p_coords in poly_coords:
            m = [1 if i < len(p_coords) else 0 for i in range(pad_size)]
            p = p_coords + [PAD_COORD for _ in range(pad_size - len(p_coords))]

            masks.append(m)
            tokens.append(p)

        return {
            "polygon": torch.tensor(tokens),
            "mask": torch.tensor(masks, dtype=torch.bool),
        }


class GraphTokenizer(Tokenizer):
    """Tokenizer for Graph-based model.

    This Tokenizer ensure the coordinates of the polygons are correctly batched,
    to be readable by a Graph-based model. Graph-based models have a specific way
    to batch data together, more information here :
    https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html

    Args:
        max_seq_len (int): Maximum sequence length. An exception will be raised if you
            try to tokenize a polygon with more points than this.
    """

    def tokenize(self, polygons):
        p_data = []
        for p in polygons:
            if isinstance(p, Polygon):
                # Don't duplicate the last coordinate, it's the same as the first one
                c = p.boundary.coords[:-1]
                # Connect the last node to the first one
                e = torch.cat([(torch.eye(2, dtype=torch.long) + i) % len(c) for i in range(len(c))], dim=-1)
            else:
                c = p.coords
                if len(c) == 1:
                    # Single node (Point). In order to avoid error, link it to itself
                    e = torch.zeros((2, 1), dtype=torch.long)
                else:
                    # Don't connect the last node to the first one
                    e = torch.cat([(torch.eye(2, dtype=torch.long) + i) % len(c) for i in range(len(c) - 1)], dim=-1)
            x = torch.tensor(c)
            p_data.append(Data(x=x, edge_index=e))

        b = Batch.from_data_list(p_data)
        return {
            "x": b.x,
            "edge_index": b.edge_index,
            "batch_index": b.batch,
        }


TOKENIZERS = {
    "transformer": TransformerTokenizer,
    "graph": GraphTokenizer,
}
