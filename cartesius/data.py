import math
import random

import torch
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import Point
import numpy as np


PAD_COORD = (0, 0)


class PolygonDataset(Dataset):
    def __init__(self, n_range, radius_range, x_min=-100, x_max=100, y_min=-100, y_max=100, tasks=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_range = n_range
        self.radius_range = radius_range
        self.tasks = tasks if tasks is not None else []

    def __len__(self):
        return 1024  # Infinite dataset

    def __getitem__(self, idx):
        x_ctr = random.randint(self.x_min, self.x_max)
        y_ctr = random.randint(self.y_min, self.y_max)
        radius = random.choice(self.radius_range)
        irregularity = random.random()
        spikeyness = random.random()
        n = random.choice(self.n_range)

        p = self._gen_poly(x_ctr, y_ctr, radius, irregularity, spikeyness, n)

        labels = [task.get_label(p) for task in self.tasks]

        return {
            "polygon": p.boundary.coords[:-1],
            "labels": labels,
        }

    def _gen_poly(self, x_ctr, y_ctr, avg_radius, irregularity, spikeyness, n_vert):
        """Method taken from https://stackoverflow.com/a/25276331

        Start with the centre of the polygon at x_ctr, y_ctr, then creates the
        polygon by sampling points on a circle around the centre. 
        Random noise is added by varying the angular spacing between sequential
        points, and by varying the radial distance of each point from the centre.

        Args:
            x_ctr (float): X coordinate of the polygon's center.
            y_ctr (float): Y coordinate of the polygon's center.
            avg_radius (float): The average radius of the polygon. This roughly
                controls how large the polygon is, really only useful for order
                of magnitude.
            irregularity (float): Parameter indicating how much variance there
                will be in the angular spacing of vertices. Should be between 0 and 1.
                [0, 1] will map to [0, 2 * pi / n_vert].
            spikeyness (float): Parameter indicating how much variance there will
                be in each vertex from the circle of radius avg_radius. [0, 1]
                will map to [0, avg_radius].
            n_vert (int): Number of vertices.

        Returns:
            shapely.geometry.Polygon: Generated Polygon.
        """
        irregularity = irregularity * 2 * math.pi / n_vert
        spikeyness = spikeyness * avg_radius

        # Generate n angle steps
        angle_steps = []
        lower = (2 * math.pi / n_vert) - irregularity
        upper = (2 * math.pi / n_vert) + irregularity
        sum = 0
        for i in range(n_vert) :
            tmp = random.uniform(lower, upper)
            angle_steps.append(tmp)
            sum = sum + tmp

        # Normalize the steps so that point 0 and point n+1 are the same
        k = sum / (2 * math.pi)
        for i in range(n_vert) :
            angle_steps[i] = angle_steps[i] / k

        # Now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(n_vert) :
            r_i = np.clip(random.gauss(avg_radius, spikeyness), 0, 2 * avg_radius)
            x = x_ctr + r_i * math.cos(angle)
            y = y_ctr + r_i * math.sin(angle)
            points.append((x, y))

            angle = angle + angle_steps[i]

        if len(points) == 1:
            return Point(points)
        elif len(points) == 2:
            return LineString(points)
        else:
            return Polygon(points)


def collate_coords(samples):
    polygons = [s["polygon"] for s in samples]

    # Collate coords by padding to the maximum length
    pad_size = max(len(p) for p in polygons)
    masks = []
    padded_polygons = []
    for poly in polygons:
        m = [1 if i < len(poly) else 0 for i in range(pad_size)]
        p = poly + [PAD_COORD for _ in range(pad_size - len(poly))]

        masks.append(m)
        padded_polygons.append(p)

    # Retrieve list of labels
    labels = [s["labels"] for s in samples]
    labels = [torch.tensor([lbl[i] for lbl in labels]) for i in range(len(labels[0]))]

    return {
        "polygon": torch.tensor(padded_polygons),
        "mask": torch.tensor(masks, dtype=torch.bool),
        "labels": labels,
    }


if __name__ == "__main__":
    from cartesius.tasks import GuessArea
    d = PolygonDataset([3, 4, 5, 6, 7, 8], [1], tasks=[GuessArea()])

    from torch.utils.data import DataLoader
    dd = DataLoader(d, batch_size=4, collate_fn=collate_coords)

    print(next(iter(dd)))
