<h1 align="center">cartesius</h1>
<p align="center">Benchmark & Pretraining for Cartesian coordinates feature extraction</p>

<p align="center"><a href="https://github.com/TeamSPWK/parka/releases"><img src="https://img.shields.io/badge/release-v0.0-blue" alt="release state" /></a>
<a href="https://github.com/TeamSPWK/cartesius/actions/workflows/pytest.yml"><img src="https://github.com/TeamSPWK/cartesius/actions/workflows/pytest.yml/badge.svg" alt="Test status" /></a>
<a href="https://github.com/TeamSPWK/cartesius/actions/workflows/lint.yml"><img src="https://github.com/TeamSPWK/cartesius/actions/workflows/lint.yml/badge.svg" alt="Lint status" /></a>
<a href="https://spwk-cartesius.readthedocs-hosted.com/en/latest/?badge=latest"><img src="https://readthedocs.com/projects/spwk-cartesius/badge/?version=latest&token=4f00a10a35b095a9d3d6284b9e16c39dee69d162b74190094b88d0583c412d0a" alt="Documentation status" /></a>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#install">Install</a> •
  <a href="#training">Training</a> •
  <a href="#testing">Testing</a> •
  <a href="#serving">Serving</a> •
  <a href="https://spwk-cartesius.readthedocs-hosted.com/en/latest/">Documentation</a> •
  <a href="#contribute">Contribute</a>
  
</p>

<h2 align="center">Description</h2>

This repository contains the code for training & benchmarking neural networks on several tasks related to feature extraction on cartesian coordinates.


<h2 align="center">Install</h2>

Clone this repository locally and install it:

```console
export GH_PAT="<your_github_pat_here>"
git clone https://$GH_PAT@github.com/TeamSPWK/cartesius.git
cd cartesius
pip install -e .
```

If you are only interested in serving, you can install it directly with `pip` :
```console
export GH_PAT="<your_github_pat_here>"
pip install git+https://$GH_PAT@github.com/TeamSPWK/cartesius.git
```

<h2 align="center">Training</h2>

🚧 WIP

<h2 align="center">Testing</h2>

🚧 WIP

<h2 align="center">Serving</h2>

🚧 WIP

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