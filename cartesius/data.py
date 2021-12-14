from functools import partial
import json
import math
import pathlib
import random

import numpy as np
import pytorch_lightning as pl
from shapely import wkt
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from cartesius.transforms import TRANSFORMS

DATA_DIR = "data"


# Monkey-patch `extract_batch_size` to not raise warning from weird tensor sizes
def extract_bs(self, batch):
    try:
        if "labels" in batch:
            batch_size = batch["labels"][0].size(0)
        else:
            batch_size = pl.utilities.data.extract_batch_size(batch)
    except RecursionError:
        batch_size = 1
    self.batch_size = batch_size
    return batch_size


pl.trainer.connectors.logger_connector.result.ResultCollection.extract_batch_size = extract_bs


class PolygonDataset(Dataset):
    """Pytorch dataset generating random Polygon.

    This dataset is used for training, as the data is randomly generated.

    Args:
        x_range (list): List of 2 elements representing the range from where to
            draw the polygon center (x-axis).
        y_range (list): List of 2 elements representing the range from where to
            draw the polygon center (y-axis).
        avg_radius_range (list): list of float, representing the possible choices
            for the average radius of the generated polygon.
        n_range (list): list of int, representing the possible choices for the number
            of points used to generate a polygon.
        tasks (list, optional): list of Tasks. These tasks will be used to compute the
            labels of each polygon. Defaults to None.
        transforms (list, optional): list of Transforms to apply to the polygons after
            they are generated and before the labels are computed. Defaults to None.
        batch_size (int, optional): Size of the batch. Defaults to 64.
        n_batch_per_epoch (int, optional): Number of batch per epoch to simulate. Since
            the dataset is infinite (randomly generated), we can choose the size of each
            epoch. Defaults to 1000.
    """

    def __init__(self,
                 x_range,
                 y_range,
                 avg_radius_range,
                 n_range,
                 tasks=None,
                 transforms=None,
                 batch_size=64,
                 n_batch_per_epoch=1000):
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.avg_radius_range = avg_radius_range
        self.n_range = n_range
        self.tasks = tasks if tasks is not None else []
        self.transforms = transforms if transforms is not None else []

        self.batch_size = batch_size
        self.n_batch_per_epoch = n_batch_per_epoch

    def __len__(self):
        return self.batch_size * self.n_batch_per_epoch  # Anyway, infinite dataset

    def __getitem__(self, idx):
        # Randomly pick parameters for polygon generation
        x_ctr = random.uniform(*self.x_range)
        y_ctr = random.uniform(*self.y_range)
        avg_radius = random.choice(self.avg_radius_range)
        irregularity = random.random()
        spikeyness = random.random()
        n = random.choice(self.n_range)

        p = self._gen_poly(x_ctr, y_ctr, avg_radius, irregularity, spikeyness, n)

        # Apply transforms
        for tr in self.transforms:
            p = tr(p)

        # Compute labels for each task
        labels = [task.get_label(p) for task in self.tasks]

        return p, labels

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
        s = 0
        for i in range(n_vert):
            tmp = random.uniform(lower, upper)
            angle_steps.append(tmp)
            s = s + tmp

        # Normalize the steps so that point 0 and point n+1 are the same
        k = s / (2 * math.pi)
        for i in range(n_vert):
            angle_steps[i] = angle_steps[i] / k

        # Now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(n_vert):
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


class PolygonTestset(Dataset):
    """Pytorch dataset reading polygons from a JSON file.

    This dataset is used for validation and testing, reading the Polygon data from a
    JSON file (since the data for validation and testing must be fixed).

    Args:
        datafile (str): Path of the JSON file containing the Polygon data
        tasks (list, optional): list of Tasks. These tasks will be used to compute the
            labels of each polygon. Defaults to None.
        transforms (list, optional): list of Transforms to apply to the polygons after
            they are generated and before the labels are computed. Defaults to None.
    """

    def __init__(self, datafile, tasks=None, transforms=None):
        super().__init__()

        self.tasks = tasks if tasks is not None else []
        self.transforms = transforms if transforms is not None else []

        # Try to load the data from local directory first
        try:
            local_datafile = pathlib.Path(__file__).parent.resolve() / DATA_DIR / datafile
            with open(local_datafile, "r", encoding="UTF-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            # Then maybe the user provided a path from the working dir ? Try to load it directly
            with open(datafile, "r", encoding="UTF-8") as f:
                data = json.load(f)

        self.polygons = [wkt.loads(d) for d in data]

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, idx):
        p = self.polygons[idx]

        # Apply transforms
        for tr in self.transforms:
            p = tr(p)

        # Compute labels for each task
        labels = [task.get_label(p) for task in self.tasks]

        return p, labels


def collate(samples, tokenizer):
    polygons = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # Tokenize the polygons
    batch = tokenizer(polygons)

    # Add the labels
    batch["labels"] = [torch.tensor([lbl[i] for lbl in labels]) for i in range(len(labels[0]))]
    return batch


class PolygonDataModule(pl.LightningDataModule):
    """DataModule for the Polygon Dataset.

    Args:
        conf (omegaconf.OmegaConf): Configuration.
        tasks (list): List of Tasks to train on.
        tokenizer (cartesius.tokenizers.Tokenizer): Tokenizer to use.
    """

    def __init__(self, conf, tasks, tokenizer):
        super().__init__()

        self.x_range = conf["x_range"]
        self.y_range = conf["y_range"]
        self.avg_radius_range = conf["avg_radius_range"]
        self.n_range = conf["n_range"]
        self.tasks = tasks
        self.transforms = [TRANSFORMS[tr](conf) for tr in conf["transforms"]]
        self.val_set_file = conf["val_set_file"]
        self.test_set_file = conf["test_set_file"]

        self.tokenizer = tokenizer
        self.collate_fn = partial(collate, tokenizer=self.tokenizer)

        self.batch_size = conf["batch_size"]
        self.n_batch_per_epoch = conf["n_batch_per_epoch"]
        self.n_workers = conf["n_workers"]

    def setup(self, stage=None):  # pylint: disable=unused-argument
        self.poly_dataset = PolygonDataset(x_range=self.x_range,
                                           y_range=self.y_range,
                                           avg_radius_range=self.avg_radius_range,
                                           n_range=self.n_range,
                                           tasks=self.tasks,
                                           transforms=self.transforms,
                                           batch_size=self.batch_size,
                                           n_batch_per_epoch=self.n_batch_per_epoch)
        self.val_dataset = PolygonTestset(datafile=self.val_set_file, tasks=self.tasks, transforms=self.transforms)
        self.test_dataset = PolygonTestset(datafile=self.test_set_file, tasks=self.tasks, transforms=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.poly_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.n_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.n_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.n_workers)
