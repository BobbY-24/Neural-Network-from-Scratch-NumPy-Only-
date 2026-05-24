# Repository Audit

## Current Purpose

I use this repository to show that I understand the mechanics of a basic neural network by implementing MNIST digit classification with NumPy only.

## Current Type

Applied ML foundations project / educational implementation.

## Problems Found

- The repository originally looked like a notebook plus raw zip file rather than a reusable implementation.
- The README described reusable source files that did not exist yet.
- The raw dataset zip was tracked in Git.
- The notebook filename was less precise than the project goal.
- There were no tests for the math functions.
- There were no methodology or limitations documents.

## Changes Made

- Renamed the notebook to `notebooks/mnist_neural_network_from_scratch.ipynb`.
- Extracted reusable NumPy implementation into `src/neural_network.py`.
- Added data helpers in `src/utils.py`.
- Added a command-line training entrypoint in `src/train.py`.
- Removed the raw zip from Git tracking and documented data setup in `data/README.md`.
- Added tests for softmax, forward-pass dimensions, finite loss, and dummy training.
- Added methodology, limitations, and results documentation.
- Rewrote the README to match the actual repository structure.

## Remaining TODOs

- Update the notebook to import the reusable `src/` implementation directly.
- Add saved training-history outputs.
- Add repeated runs and baseline comparisons.
- Add robustness experiments with noisy or transformed digits.
