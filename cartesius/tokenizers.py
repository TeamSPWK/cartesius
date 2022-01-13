from shapely.geometry import Polygon
import torch

PAD_COORD = (0, 0)


class Tokenizer:
    """Basic Tokenizer.

    Tokenizers takes as input a polygon or a batch of polygons and return a
    dict of tensor representing the polygon(s).

    Sub classes should overwrite the method `tokenize()`.

    Args:
        max_seq_len (int, optional): Maximum sequence length. An exception will be raised if you
            try to tokenize a polygon with more points than this. Defaults to 256.
    """

    def __init__(self, *args, max_seq_len=256, **kwargs):  # pylint: disable=unused-argument
        self.max_seq_len = max_seq_len

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
