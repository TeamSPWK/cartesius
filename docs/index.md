# Cartesius

## Introduction

Welcome to the documentation of the `cartesius` package.

This package contains the data for **training & benchmarking** neural networks on various tasks, with the goal to evaluate **feature extraction** capabilities of benchmarked models.

## Installation

### Latest version

You can install `cartesius` with :

```bash
pip install spwk-cartesius
```

### Local

You can also clone the repository locally and install it manually :

```bash
git clone https://$GH_PAT@github.com/TeamSPWK/cartesius.git
cd cartesius
pip install -e .
```

### Extra dependencies

You can also install extras dependencies, for example :

```bash
pip install spwk-cartesius[docs]
```

Will install necessary dependencies for building the docs.

!!! hint
    If you installed the package locally, use :
    ```bash
    pip install -e .[docs]
    ```

---

List of extra dependencies :

* **`docs`** : Dependencies for building documentation.
* **`tests`** : Dependencies for running unit-tests.
* **`lint`** : Dependencies for running linters & formatters.
* **`dev`** : `docs` + `tests` + `lint`.
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
