import numpy as np
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import Polygon
import torch

from cartesius.converters import AugmentConverter
from cartesius.converters import pad_arr
from cartesius.converters import PolarConverter

try:
    # Optional, have to install additional dependencies for this to work
    import cv2
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
    to batch data together, more information
    [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)
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


class TransformerAugmentTokenizer(Tokenizer):
    """Tokenizer for Transformer model, with vertex augmentation.

    This Tokenizer augment / remove some vertices to certain sequence length.
    When we need to augment vertices, it generates priority indices with edge length,
    and assigns number of vertices.
    When we need to remove vertices, it generates priority indices with cosine value,
    and itertively removes vertices.

    Args:
        max_seq_len (int): Maximum sequence length. An exception will be raised if you
            try to tokenize a polygon with more points than this.
    """

    def __init__(self, max_seq_len, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.max_seq_len = max_seq_len
        self.augment_converter = AugmentConverter(max_seq_len)

    def tokenize_each(self, p):
        if isinstance(p, Polygon):
            x_arr, y_arr = [np.array(x) for x in p.boundary.xy]
        else:
            x_arr, y_arr = [np.array(x) for x in p.xy]
        new_arr = np.stack(self.augment_converter(x_arr, y_arr), axis=-1)
        return new_arr

    def tokenize(self, polygons):
        x = torch.tensor([self.tokenize_each(p) for p in polygons], dtype=torch.float32)
        return {
            "polygon": x,
            "mask": torch.ones((x.shape[:-1]), dtype=torch.bool),
        }


class TransformerPolarTokenizer(Tokenizer):
    """Tokenizer for Transformer model, with Polar coordinates.

    This Tokenizer transforms Cartesian coordinates into polar coordinates.

    Its polygon has 2 features : theta, r.
    theta is the angle from a reference direction (1, 0).
    r is the distance from a reference point (0, 0).

    Args:
        max_seq_len (int): Maximum sequence length. An exception will be raised if you
            try to tokenize a polygon with more points than this.
    """

    def __init__(self, max_seq_len, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.max_seq_len = max_seq_len
        self.polar_converter = PolarConverter()

    def tokenize(self, polygons):
        x_arrs = []
        y_arrs = []
        for p in polygons:
            if isinstance(p, Polygon):
                x_arr, y_arr = [np.array(x) for x in p.boundary.xy]
            else:
                x_arr, y_arr = [np.array(x) for x in p.xy]
            x_arrs.append(x_arr)
            y_arrs.append(y_arr)
        pad_size = max(len(x_arr) for x_arr in x_arrs)
        if pad_size > self.max_seq_len:
            raise RuntimeError(f"Polygons are too big to be tokenized ({pad_size} > {self.max_seq_len})")
        masks = []
        tokens = []
        for x_arr, y_arr in zip(x_arrs, y_arrs):
            m = [1 if i < len(x_arr) else 0 for i in range(pad_size)]
            x_arr, y_arr = [pad_arr(x, pad_size) for x in (x_arr, y_arr)]
            theta_arr, r_arr = self.polar_converter(x_arr, y_arr)
            masks.append(m)
            token = np.stack([theta_arr, r_arr], axis=-1)
            tokens.append(token)
        return {
            "polygon": torch.tensor(tokens, dtype=torch.float32),
            "mask": torch.tensor(masks, dtype=torch.bool),
        }


class TransformerCartePolarTokenizer(Tokenizer):
    """Tokenizer for Transformer model, with both Cartesian and Polar coordinates.

    This Tokenizer transforms Cartesian coordinates into polar coordinates,
    and aggregates them with original Catresian coordinate information.

    Its polygon feature has 4 columns : x, y, theta, r.
    x is the sequence of x coordinates of the vertices
    y is the sequence of y coordinates of the vertices
    theta is the sequence of angles of the vertices from a reference direction (1, 0).
    r is the sequence of distances of the vertices from a reference point (0, 0).

    Args:
        max_seq_len (int): Maximum sequence length. An exception will be raised if you
            try to tokenize a polygon with more points than this.
    """

    def __init__(self, max_seq_len, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.carte_tokenizer = TransformerTokenizer(max_seq_len)
        self.polar_tokenizer = TransformerPolarTokenizer(max_seq_len)

    def tokenize(self, polygons):
        carte_dict = self.carte_tokenizer.tokenize(polygons)
        polar_dict = self.polar_tokenizer.tokenize(polygons)
        carte_polar_dict = {
            "polygon": torch.cat([carte_dict["polygon"], polar_dict["polygon"]], -1),
            "mask": carte_dict["mask"]
        }
        return carte_polar_dict


class GraphPolarTokenizer(Tokenizer):
    """Tokenizer for Graph-based model, with Polar coordinates.

    This Tokenizer transforms Cartesian coordinates into polar coordinates,
    and generate tokens that is readable for Graph-based model.

    Its x feature has 2 columns : theta, r.
    theta is the sequence of angles of the vertices from a reference direction (1, 0).
    r is the sequence of distances of the vertices from a reference point (0, 0).
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.polar_converter = PolarConverter()

    def tokenize(self, polygons):
        p_data = []
        for p in polygons:
            if isinstance(p, Polygon):
                x_arr, y_arr = [np.array(x)[:-1] for x in p.boundary.xy]
            else:
                x_arr, y_arr = [np.array(x) for x in p.xy]
            r_arr, theta_arr = self.polar_converter(x_arr, y_arr)
            c = np.stack([r_arr, theta_arr], axis=-1)
            if isinstance(p, Polygon):
                e = torch.cat([(torch.eye(2, dtype=torch.long) + i) % len(c) for i in range(len(c))], dim=-1)
            else:
                if len(c) == 1:
                    e = torch.zeros((2, 1), dtype=torch.long)
                else:
                    e = torch.cat([(torch.eye(2, dtype=torch.long) + i) % len(c) for i in range(len(c) - 1)], dim=-1)
            x = torch.tensor(c, dtype=torch.float32)
            p_data.append(Data(x=x, edge_index=e))

        b = Batch.from_data_list(p_data)
        return {
            "x": b.x,
            "edge_index": b.edge_index,
            "batch_index": b.batch,
        }


class GraphCartePolarTokenizer(Tokenizer):
    """Tokenizer for Transformer model, with both Cartesian and Polar coordinates.

    This Tokenizer transforms Cartesian coordinates into polar coordinates,
    and aggregates them with original Catresian coordinate information.

    Its x feature has 4 columns : x, y, theta, r.
    x is the sequence of x coordinates of the vertices
    y is the sequence of y coordinates of the vertices
    theta is the sequence of angles of the vertices from a reference direction (1, 0).
    r is the sequence of distances of the vertices from a reference point (0, 0).
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.carte_tokenizer = GraphTokenizer()
        self.polar_tokenizer = GraphPolarTokenizer()

    def tokenize(self, polygons):
        carte_dict = self.carte_tokenizer.tokenize(polygons)
        polar_dict = self.polar_tokenizer.tokenize(polygons)
        carte_polar_dict = {
            "x": torch.cat([carte_dict["x"], polar_dict["x"]], -1),
            "edge_index": carte_dict["edge_index"],
            "batch_index": carte_dict["batch_index"],
        }
        return carte_polar_dict


class RasterTokenizer(Tokenizer):
    """Tokenizer for Image-based model.

    This tokenizer generates image with canvas size 128 * 128,
    which colors inside of polygon with 1 and outside of polygon with 0.
    Shape of returned tensor is (batch size, 3, canvas length, canvas length).
    """

    def tokenize(self, polygons):
        canvas_len = 128
        poly_max_len = 1
        canvas_center = np.array((canvas_len, canvas_len)) / 2

        batch = []
        for p in polygons:
            canvas = np.zeros((canvas_len, canvas_len, 3), dtype=np.uint8)
            if isinstance(p, Polygon):
                poly_pts = np.array(p.boundary.coords)[:-1]
            else:
                poly_pts = np.array(p.coords)
            centralized_poly = ((poly_pts - np.array(box(*p.bounds).centroid.coords)) / (poly_max_len / canvas_len) +
                                canvas_center)
            preprocessed_poly = centralized_poly.astype(np.int32).reshape(-1, 1, 2)
            if isinstance(p, LineString):
                img = cv2.polylines(canvas, [preprocessed_poly], False, (255, 255, 255)) / 255
            else:
                img = cv2.fillPoly(canvas, [preprocessed_poly], (255, 255, 255)) / 255
            batch.append(img)
        batch = np.rollaxis(np.stack(batch), -1, 1)
        x = torch.tensor(batch, dtype=torch.float32)
        return {"x": x}


TOKENIZERS = {
    "transformer": TransformerTokenizer,
    "transformer_augment": TransformerAugmentTokenizer,
    "transformer_polar": TransformerPolarTokenizer,
    "transformer_carte_polar": TransformerCartePolarTokenizer,
    "graph": GraphTokenizer,
    "graph_polar": GraphPolarTokenizer,
    "graph_carte_polar": GraphCartePolarTokenizer,
    "raster": RasterTokenizer
}