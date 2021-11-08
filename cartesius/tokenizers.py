import torch


PAD_COORD = (0, 0)


class Tokenizer:
    """Base class for Tokenizer.

    Tokenizers takes as input a polygon or a batch of polygons and return a
    dict of tensor representing the polygon(s).

    Sub classes should overwrite the method `tokenize()`.
    """

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
    def tokenize(self, polygons):
        pad_size = max(len(p) for p in polygons)
        masks = []
        tokens = []
        for poly in polygons:
            m = [1 if i < len(poly) else 0 for i in range(pad_size)]
            p = poly + [PAD_COORD for _ in range(pad_size - len(poly))]

            masks.append(m)
            tokens.append(p)

        return {
            "polygon": torch.tensor(tokens),
            "mask": torch.tensor(masks, dtype=torch.bool),
        }


TOKENIZERS = {
    "transformer": TransformerTokenizer,
}
