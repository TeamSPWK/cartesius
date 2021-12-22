import numpy as np
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


class AugmentTokenizer(Tokenizer):
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

    @staticmethod
    def remove_vertices(arr, edge_vectors, req_vertices, tolerance):
        if np.linalg.norm(arr[-1] - arr[0]) < tolerance:
            next_vectors = edge_vectors / np.expand_dims(np.linalg.norm(edge_vectors, axis=-1), axis=-1) + 1e-4
            prev_vectors = -np.roll(edge_vectors, 1, axis=0)
            cosine = np.dot(next_vectors, prev_vectors.T).diagonal()
            priority_indices = np.argsort(-cosine)
            adjusted_arr = np.delete(arr[:-1], priority_indices[req_vertices:], axis=0)
            adjusted_arr = np.concatenate([adjusted_arr, adjusted_arr[:1]])
        else:
            next_norms = np.expand_dims(np.linalg.norm(edge_vectors[1:], axis=-1), axis=-1) + 1e-4
            next_vectors = edge_vectors[1:] / next_norms
            prev_norms = np.expand_dims(np.linalg.norm(edge_vectors[:-1], axis=-1), axis=-1) + 1e-4
            prev_vectors = -edge_vectors[:-1] / prev_norms
            cosine = np.dot(next_vectors, prev_vectors.T).diagonal()
            cosine = np.concatenate([np.array([0]), cosine, np.array([0])])
            priority_indices = np.argsort(-cosine)
            adjusted_arr = np.delete(arr, priority_indices[req_vertices:], axis=0)
        return adjusted_arr

    @staticmethod
    def augment_vertices(arr, edge_vectors, req_vertices):
        edge_lengths = np.linalg.norm(edge_vectors, axis=-1)
        req_vertices_per_edges = req_vertices * (edge_lengths / np.sum(edge_lengths) + 1e-4)
        req_vertices_per_edges_decimal = req_vertices_per_edges - np.floor(req_vertices_per_edges)
        rounded_req_vertices_per_edges = np.floor(req_vertices_per_edges).astype(np.int)
        deficient_req_vertices = req_vertices - np.sum(rounded_req_vertices_per_edges)
        priority_indices = np.argsort(-req_vertices_per_edges_decimal)
        for i in range(deficient_req_vertices):
            rounded_req_vertices_per_edges[priority_indices[i % len(priority_indices)]] += 1
        rounded_req_edges_per_edges = rounded_req_vertices_per_edges + 1
        interpolate_vectors = edge_vectors / np.expand_dims((rounded_req_edges_per_edges), axis=-1) + 1e-4
        vertices_list = []
        for idx, (interpolate_vector,
                  rounded_req_edges_per_edge) in enumerate(zip(interpolate_vectors, rounded_req_edges_per_edges)):
            vertices_list.append(arr[idx] +
                                 interpolate_vector * np.expand_dims(np.arange(rounded_req_edges_per_edge), axis=-1))
        adjusted_arr = np.concatenate(vertices_list + [arr[-1:]])
        return adjusted_arr

    def tokenize_each(self, p, tolerance, max_seq_len):
        if isinstance(p, Polygon):
            arr = np.array(p.boundary.coords[:-1])
        else:
            arr = np.array(p.coords)
        if (arr == 0).all():
            adjusted_arr = np.zeros((max_seq_len, 2))
            return adjusted_arr
        if len(arr) < 2:
            adjusted_arr = np.zeros((max_seq_len, 2))
            return adjusted_arr
        req_vertices = max_seq_len - len(arr)
        edge_vectors = arr[1:] - arr[:-1]
        if np.linalg.norm(arr[-1] - arr[0]) < tolerance:
            req_vertices += 1
        if req_vertices == 0:
            adjusted_arr = arr
        elif req_vertices < 0:
            adjusted_arr = self.remove_vertices(arr, edge_vectors, req_vertices, tolerance)
        else:
            adjusted_arr = self.augment_vertices(arr, edge_vectors, req_vertices)
        if np.linalg.norm(adjusted_arr[-1] - adjusted_arr[0]) < 1e-4:
            adjusted_arr = adjusted_arr[:-1]
        return adjusted_arr

    def tokenize(self, polygons):
        tolerance = 1e-4
        x = torch.tensor([self.tokenize_each(p, tolerance, self.max_seq_len) for p in polygons], dtype=torch.float32)
        return {
            "polygon": x,
            "mask": torch.ones((x.shape[:-1]), dtype=torch.bool),
        }


class PolarTokenizer(Tokenizer):
    """Tokenizer for Transformer model, with Polar coordinates

    This Tokenizer transforms Cartesian coordinates into polar coordinates.
    r is the distance from a reference point (0, 0).
    theta is the angle from a reference direction (1, 0).

    Args:
        max_seq_len (int): Maximum sequence length. An exception will be raised if you
            try to tokenize a polygon with more points than this.
    """

    def __init__(self, max_seq_len, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()

        self.max_seq_len = max_seq_len

    @staticmethod
    def tokenize_each(p_coords):
        arr = np.array(p_coords)
        r = np.linalg.norm(arr, axis=-1)
        theta = np.arctan2(arr[:, 1], arr[:, 0])
        return r, theta

    @staticmethod
    def pad(arr, pad_size):
        return np.pad(arr, (0, pad_size - len(arr)), mode="constant", constant_values=0)

    def tokenize(self, polygons):
        poly_coords = [list(p.boundary.coords[:-1]) if isinstance(p, Polygon) else list(p.coords) for p in polygons]
        pad_size = max(len(p_coords) for p_coords in poly_coords)
        if pad_size > self.max_seq_len:
            raise RuntimeError(f"Polygons are too big to be tokenized ({pad_size} > {self.max_seq_len})")
        masks = []
        tokens = []
        for p_coords in poly_coords:
            m = [1 if i < len(p_coords) else 0 for i in range(pad_size)]
            r, theta = [self.pad(x, pad_size) for x in self.tokenize_each(p_coords)]
            token = np.stack([r, theta], axis=-1)
            masks.append(m)
            tokens.append(token)
        return {
            "polygon": torch.tensor(tokens, dtype=torch.float32),
            "mask": torch.tensor(masks, dtype=torch.bool),
        }


TOKENIZERS = {
    "transformer": TransformerTokenizer,
    "graph": GraphTokenizer,
    "augment": AugmentTokenizer,
    "polar": PolarTokenizer
}
