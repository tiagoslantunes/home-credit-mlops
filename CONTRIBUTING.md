# Contributing

This repository is a portfolio fork of the collaborative
[upstream project](https://github.com/marianamelo0/home-credit-mlops). Check the
upstream repository before starting substantial work and keep authorship and
project provenance intact.

## Development setup

On Windows, run `./setup.ps1`. On other platforms, create a Python 3.12 virtual
environment, install `requirements.txt`, and then install the project with
`python -m pip install -e ".[dev]"`.

## Before opening a pull request

1. Create a focused branch from `main`.
2. Do not commit Kaggle data, credentials, trained models, MLflow stores or
   generated reports.
3. Run `python -m compileall -q app src tests`.
4. Validate notebook JSON and run the relevant tests.
5. Explain the behaviour change, validation performed and any data or model
   impact in the pull request.

Changes to model features, thresholds, evaluation or serving contracts should
also update `MODEL_CARD.md` and the corresponding configuration under `conf/`.
