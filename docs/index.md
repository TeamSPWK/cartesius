# Cartesius

## Introduction

Welcome to the documentation of the `cartesius` package.

This package contains the code for **training & benchmarking** neural networks on several tasks related to **feature extraction** on cartesian coordinates.

## Installation

### Latest version

You can install the latest `cartesius` with :

```bash
export GH_PAT="<your_github_pat_here>"
pip install git+https://$GH_PAT@github.com/TeamSPWK/cartesius.git
```

!!! question "Why do we need to set GH_PAT ?"
    Because the repository is **private**, we need to be **authenticated** to access the repository. A _Github Personal Access Token_ is the standard way to do this.

### Specific version

Alternatively, you can install a specific version (`v0.1` in this example) with :

```bash
export GH_PAT="<your_github_pat_here>"
pip install git+https://$GH_PAT@github.com/TeamSPWK/cartesius.git@v0.1
```

### Local

You can also clone the repository locally and install it manually :

```bash
export GH_PAT="<your_github_pat_here>"
git clone https://$GH_PAT@github.com/TeamSPWK/cartesius.git
cd cartesius
pip install -e .
```

### Extra dependencies

You can also install extras dependencies, for example :

```bash
pip install -e .[docs]
```

Will install necessary dependencies for building the docs.

!!! hint
    If you installed the package directly from github :
    ```bash
    pip install "cartesius[docs] @ git+https://$GH_PAT@github.com/TeamSPWK/cartesius.git"
    ```

---

List of extra dependencies :

* **`docs`** : Dependencies for building documentation.
* **`tests`** : Dependencies for running unit-tests.
* **`lint`** : Dependencies for running linters & formatters.
* **`dev`** : `docs` + `tests` + `lint`.
* **`graph`** : Dependencies for graph-based models.  
_Note that you need to install `torch-scatter` and `torch-sparse` manually. See [Graph-based : Installation](models/graph.md#installation) for more details._
* **`all`** : All extra dependencies.

## Contribute

To contribute, install the package locally (see [Installation](#local)), create your own branch, add your code/tests/documentation, and open a PR !

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
