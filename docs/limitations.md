# Limitations

## Current Weaknesses

- This is an educational implementation, not a production deep learning framework.
- The training loop uses full-batch gradient descent and is not optimized for speed.
- I have not added repeated runs, confidence intervals, or systematic hyperparameter tuning.
- The notebook still needs to be updated to import the reusable `src/` code.

## Dataset Limitations

- MNIST-style digit data is clean and benchmark-like.
- It does not test robustness to distribution shift, rotations, noise, handwriting variation, or adversarial perturbations.

## Modeling Limitations

- The architecture is intentionally small.
- I do not compare against logistic regression, scikit-learn baselines, or PyTorch yet.
- I do not save trained weights or training history artifacts yet.

## Future Improvements

- Add baseline comparisons.
- Add mini-batch gradient descent.
- Add model checkpoint saving.
- Add tests for gradient shapes and numerical stability.
- Add robustness experiments on perturbed images.
