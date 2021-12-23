import numpy as np


class Converter:
    """Base class for converters. A Converter is a callable that take a sequence array of x, and y
    as input and convert into learnable features.
    """

    def __call__(self, x_arr, y_arr):
        """Main method of the Converter, it takes as input sequence array of x, and y to
        convert, and return the learnable features.

        Args:
            x_arr (numpy.ndrray): Sequence array of the x coordinates.
            y_arr (numpy.ndrray): Sequence array of the y coordinates.

        Raises:
            NotImplementedError: Exception raised if this method is not
                overwritten by the subclass.
        """
        raise NotImplementedError


class PolarConverter(Converter):
    """Converter that converts the sequence array of x, y coordinates into polar coodinates.
    """

    def __init__(self):
        pass

    def __call__(self, x_arr, y_arr):
        """Method that converts sequence array of x, y coordinates into polar coordinates.

        Args:
            x_arr (np.ndarray): Sequence array of the x coordinates.
            y_arr (np.ndarray): Sequence array of the y coordinates.

        Returns:
            (tuple): tuple containing:
                (np.ndarray): Sequence array of the distance from a reference point (0, 0).
                (np.ndarray): Sequence array of the angle from a reference direction (1, 0).
        """
        arr = np.stack([x_arr, y_arr], axis=-1)
        r_arr = np.linalg.norm(arr, axis=-1)
        theta_arr = np.arctan2(y_arr, x_arr)
        return r_arr, theta_arr


class AugmentConverter(Converter):
    """Converter that augments the vertices to fixed length with adding and removing vertices.
    """

    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    @staticmethod
    def remove_vertices(x_arr, y_arr, req_vertices):
        """Method that removes vertices to required number of vertices.

        It iteratively removes vertices with respect to cosine of vertex internal angles.
        The cosine of vertex internal angle larger, the priority of vertex bigger.
        It removes vertices from smaller priority, to bigger priority.

        Args:
            x_arr (np.ndarray): Sequence array of the x coordinates.
            y_arr (np.ndarray): Sequence array of the y coordinates.
            req_vertices (int): Required number of vertices

        Returns:
            tuple: tuple containing:
                np.ndarray: Concise sequence array of the x coordinates.
                np.ndarray: Concise sequence array of the y coordinates.
        """
        arr = np.stack([x_arr, y_arr], axis=-1)
        edge_vectors = arr[1:] - arr[:-1]
        next_norms = np.expand_dims(np.linalg.norm(edge_vectors[1:], axis=-1), axis=-1) + 1e-9
        next_vectors = edge_vectors[1:] / next_norms
        prev_norms = np.expand_dims(np.linalg.norm(edge_vectors[:-1], axis=-1), axis=-1) + 1e-9
        prev_vectors = -edge_vectors[:-1] / prev_norms
        cosine = np.dot(next_vectors, prev_vectors.T).diagonal()
        cosine = np.concatenate([np.array([0]), cosine, np.array([0])])
        priority_indices = np.argsort(-cosine)
        adjusted_arr = np.delete(arr, priority_indices[req_vertices:], axis=0)
        new_x_arr, new_y_arr = adjusted_arr[:, 0], adjusted_arr[:, 1]
        return new_x_arr, new_y_arr

    @staticmethod
    def augment_vertices(x_arr, y_arr, req_vertices):
        """Method that augments vertices to required number of vertices.

        It iteratively add vertices with respect to length of edges
        The length of edge longer, the priority of vertex bigger.
        It adds vertices from bigger priority, to smaller priority.

        Args:
            x_arr (np.ndarray): Sequence array of the x coordinates.
            y_arr (np.ndarray): Sequence array of the y coordinates.
            req_vertices (int): Required number of vertices

        Returns:
            tuple: tuple containing:
                np.ndarray: Augmented sequence array of the x coordinates.
                np.ndarray: Augmented sequence array of the y coordinates.
        """
        arr = np.stack([x_arr, y_arr], axis=-1)
        edge_vectors = arr[1:] - arr[:-1]
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
        new_x_arr, new_y_arr = adjusted_arr[:, 0], adjusted_arr[:, 1]
        return new_x_arr, new_y_arr

    def __call__(self, x_arr, y_arr):
        """Method that augments, or removes sequence array of x, y coordinates.

        Args:
            x_arr (np.ndarray): Sequence array of the x coordinates.
            y_arr (np.ndarray): Sequence array of the y coordinates.

        Returns:
            tuple: tuple containing:
                np.ndarray: Augmented sequence array of the x coordinates.
                np.ndarray: Augmented sequence array of the y coordinates.
        """
        arr = np.stack([x_arr, y_arr], axis=-1)
        if (arr == 0).all() or len(arr) < 2:
            return np.zeros(self.max_seq_len), np.zeros(self.max_seq_len)
        req_vertices = self.max_seq_len - len(arr)
        if np.linalg.norm(arr[-1] - arr[0]) == 0:
            repeated_last_vertex = True
            x_arr, y_arr = x_arr[:-1], y_arr[:-1]
        else:
            repeated_last_vertex = False
        if req_vertices > 0:
            new_x_arr, new_y_arr = self.augment_vertices(x_arr, y_arr, req_vertices)
        elif req_vertices < 0:
            new_x_arr, new_y_arr = self.remove_vertices(x_arr, y_arr, req_vertices)
        else:
            new_x_arr, new_y_arr = arr[:, 0], arr[:, 1]
        if repeated_last_vertex:
            new_x_arr = np.concatenate([new_x_arr, new_x_arr[:1]])
            new_y_arr = np.concatenate([new_y_arr, new_y_arr[:1]])
        return new_x_arr, new_y_arr


def pad_arr(arr, pad_size, pad_value=0):
    """Padding function for array

    Args:
        arr (np.ndarray): 1-dimensional array to be padded
        pad_size (int): Size to be padded
        pad_value (int, optional): Padding value. Defaults to 0.

    Returns:
        np.ndarray: Padded array
    """
    return np.pad(arr, (0, pad_size - len(arr)), mode="constant", constant_values=pad_value)
