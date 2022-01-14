<h1 align="center">cartesius</h1>
<p align="center">Benchmark for Cartesian coordinates feature extraction</p>

<p align="center">
    <a href="https://github.com/TeamSPWK/cartesius/releases"><img src="https://img.shields.io/github/release/TeamSPWK/cartesius.svg" alt="GitHub release" /></a>
    <a href="https://github.com/TeamSPWK/cartesius/actions/workflows/pytest.yml"><img src="https://github.com/TeamSPWK/cartesius/actions/workflows/pytest.yml/badge.svg" alt="Test status" /></a>
    <a href="https://github.com/TeamSPWK/cartesius/actions/workflows/lint.yml"><img src="https://github.com/TeamSPWK/cartesius/actions/workflows/lint.yml/badge.svg" alt="Lint status" /></a>
    <a href="https://spwk-cartesius.readthedocs-hosted.com/en/latest/?badge=latest"><img src="https://readthedocs.com/projects/spwk-cartesius/badge/?version=latest&token=23bd7924365dc7d2aecf8f3af3bdd2bfd045d1a17674a28bf3d857c3a6afef97" alt="Documentation status" /></a>
    <a href="https://www.kaggle.com/c/cartesius/"><img src="https://img.shields.io/badge/kaggle-cartesius-blueviolet" alt="Kaggle" /></a>
</p>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="https://spwk-cartesius.readthedocs-hosted.com/en/latest/">Documentation</a> •
  <a href="#contribute">Contribute</a>
  
</p>

<h2 align="center">Description</h2>

This repository contains the data for training & benchmarking neural networks on various tasks, with the goal to evaluate **feature extraction** capabilities of benchmarked models.

Extracting features 2D polygons is not a trivial task. Many models can be applied to this task, and many approaches exist (learning from raw coordinates, learning from a raster image, etc...).

So it's necessary to have a benchmark, in order to quantify and see which model/approach is the best.


<h2 align="center">Install</h2>

Install `cartesius` by running :

```bash
pip install spwk-cartesius
```

<h2 align="center">Usage</h2>

In `cartesius`, the training data is polygons that are randomly generated.

Let's have a look. First, initialize the training set :

```python
from cartesius.data import PolygonDataset

train_data = PolygonDataset(
    x_range=[-50, 50],          # Range for the center of the polygon (x)
    y_range=[-50, 50],          # Range for the center of the polygon (y)
    avg_radius_range=[1, 10],   # Average radius of the generated polygons. Here it will either generate polygons with average radius 1, or 10
    n_range=[6, 8, 11],         # Number of points in the polygon. here it will either generate polygons with 6, 8 or 11 points
)
```

Then, we will take a look at the generated polygon :

```python
import matplotlib.pyplot as plt
from cartesius.utils import print_polygon

def disp(*polygons):
    plt.clf()
    for p in polygons:
      print_polygon(p)
    plt.gca().set_aspect(1)
    plt.axis("off")
    plt.show()

polygon, labels = train_data[0]
disp(polygon)
print(labels)
```

---

The benchmark relies on various tasks : predicting the **area** of a polygon, its **perimeter**, its **centroid**, etc... (see the documentation for more details)

The goal of the benchmark is to write an **encoder** : a model that can encode a polygon's features into a vector.

After the feature vector is extracted from the polygon using the encoder, several heads (one per task) will predict the labels. If the polygon is well represented through the extracted features, the task-heads should have no problem predicting the labels.

---

The `notebooks/` folder contains a notebook that implements a Transformer model, trains it on `cartesius` data, and evaluate it. You can use this notebook as a model for further research.

_Note : At the end of the notebook, a file `submission.csv` is saved, you can use it for the [Kaggle competition](https://www.kaggle.com/c/cartesius/)._

<h2 align="center">Contribute</h2>

To contribute, install the package locally, create your own branch, add your code/tests/documentation, and open a PR !

### Unit tests

When you add some feature, you should add tests for it and ensure the previous tests pass :

```bash
python -m pytest -W ignore::DeprecationWarning
```

### Linters & formatters

Your code should be linted and properly formatted :

```bash
isort . && yapf -ri . && pylint cartesius && pylint tests --disable=redefined-outer-name
```

### Documentation

The documentation should be kept up-to-date. You can visualize the documentation locally by running :

```bash
mkdocs serve
```