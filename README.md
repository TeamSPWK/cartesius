<h1 align="center">cartesius</h1>
<p align="center">Benchmark & Pretraining for Cartesian coordinates feature extraction</p>

<p align="center"><a href="https://github.com/TeamSPWK/parka/releases"><img src="https://img.shields.io/badge/release-v0.0-blue" alt="release state" /></a>
<a href="https://github.com/TeamSPWK/cartesius/actions/workflows/pytest.yml"><img src="https://github.com/TeamSPWK/cartesius/actions/workflows/pytest.yml/badge.svg" alt="Test status" /></a>
<a href="https://github.com/TeamSPWK/cartesius/actions/workflows/lint.yml"><img src="https://github.com/TeamSPWK/cartesius/actions/workflows/lint.yml/badge.svg" alt="Lint status" /></a>
<a href="https://spwk-cartesius.readthedocs-hosted.com/en/latest/?badge=latest"><img src="https://readthedocs.com/projects/spwk-cartesius/badge/?version=latest&token=23bd7924365dc7d2aecf8f3af3bdd2bfd045d1a17674a28bf3d857c3a6afef97" alt="Documentation status" /></a>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="https://spwk-cartesius.readthedocs-hosted.com/en/latest/">Documentation</a> •
  <a href="#contribute">Contribute</a>
  
</p>

<h2 align="center">Description</h2>

This repository contains the code for training & benchmarking neural networks on several tasks related to feature extraction on cartesian coordinates.


<h2 align="center">Install</h2>

Install `cartesius` by running :

```bash
export GH_PAT="<your_github_pat_here>"
pip install git+https://$GH_PAT@github.com/TeamSPWK/cartesius.git
```

---

For development, you can install it locally by first cloning the repository :

```bash
export GH_PAT="<your_github_pat_here>"
git clone https://$GH_PAT@github.com/TeamSPWK/cartesius.git
cd cartesius
pip install -e .
```

<h2 align="center">Usage</h2>

Just run the command `cartesius`, it will train and test a model with the default configuration (located in `cartesius/config/default.yaml`).

---

You can change the configuration by specifying a different configuration file :

```bash
cartesius config=transformer.yaml
```

---

You can also change each value of the configuration independently, directly from the command line :

```bash
cartesius seed=666 activation=relu
```

---

You can test a specific checkpoint by running :

```bash
cartesius train=False test=True ckpt=<path/to/my/model.ckpt>
```

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